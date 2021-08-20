/*
 * Copyright (c) 2017-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/functions/CLFFTConvolutionLayer.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/gpu/cl/operators/ClConv2d.h"
#include "support/Cast.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::experimental;
struct CLConvolutionLayer::Impl
{
    MemoryGroup                          memory_group{};
    std::shared_ptr<IMemoryManager>      memory_manager{};
    std::unique_ptr<opencl::IClOperator> op{ nullptr };
    ITensorPack                          run_pack{};
    ITensorPack                          prep_pack{};
    WorkspaceData<CLTensor>              workspace{};
    experimental::MemoryRequirements     aux_mem_req{};
    std::unique_ptr<IFunction>           func{ nullptr };
};

CLConvolutionLayer::CLConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_manager = std::move(memory_manager);
}

CLConvolutionLayer::~CLConvolutionLayer() = default;

void CLConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                   const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, weights_info, dilation, act_info, enable_fast_math, num_groups);
}

void CLConvolutionLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                   const WeightsInfo &weights_info,
                                   const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLConvolutionLayer::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info, weights_info, dilation, act_info,
                                                            enable_fast_math, num_groups));

    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, enable_fast_math, num_groups);

    switch(opencl::ClConv2d::get_convolution_method(input->info(), weights->info(), output->info(), conv2d_info,
                                                    weights_info, CLScheduler::get().target()))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::DIRECT:
        case ConvolutionMethod::GEMM:
        {
            auto f = std::make_unique<opencl::ClConv2d>();
            f->configure(compile_context, input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv2d_info, weights_info);
            _impl->op = std::move(f);
            break;
        }
        case ConvolutionMethod::FFT:
        {
            auto f = std::make_unique<CLFFTConvolutionLayer>(_impl->memory_manager);
            f->configure(compile_context, input, weights, biases, output, conv_info, act_info, enable_fast_math);
            _impl->func = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    if(_impl->op)
    {
        _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
        _impl->aux_mem_req  = _impl->op->workspace();
        _impl->run_pack     = { { ACL_SRC_0, input }, { ACL_SRC_1, weights }, { ACL_SRC_2, biases }, { ACL_DST, output } };
        _impl->prep_pack    = { { ACL_SRC_1, weights }, { ACL_SRC_2, biases } };
        _impl->workspace    = manage_workspace<CLTensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
    }
}

Status CLConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                    const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((num_groups != 1) && (input->data_layout() != DataLayout::NCHW), "Grouping (num_groups != 1) with NHWC data layout is not supported");

    const GPUTarget  gpu_target  = CLScheduler::get().target();
    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, enable_fast_math, num_groups);

    switch(opencl::ClConv2d::get_convolution_method(input, weights, output, conv2d_info, weights_info, gpu_target))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::DIRECT:
        case ConvolutionMethod::GEMM:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(opencl::ClConv2d::validate(input, weights, biases, output, conv2d_info, weights_info));
            break;
        }
        case ConvolutionMethod::FFT:
        {
            // Validate FFT-based convolution layer
            ARM_COMPUTE_RETURN_ON_ERROR(CLFFTConvolutionLayer::validate(input, weights, nullptr, output, conv_info, act_info, enable_fast_math));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

ConvolutionMethod CLConvolutionLayer::get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                             const WeightsInfo &weights_info, const ActivationLayerInfo &act_info, const GPUTarget gpu_target, const Size2D &dilation, bool enable_fast_math)
{
    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, enable_fast_math, 1);
    return opencl::ClConv2d::get_convolution_method(input, weights, output, conv2d_info, weights_info, gpu_target);
}

void CLConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    if(_impl->func)
    {
        _impl->func->run();
    }
    else
    {
        _impl->op->run(_impl->run_pack);
    }
}

void CLConvolutionLayer::prepare()
{
    if(_impl->func)
    {
        _impl->func->prepare();
    }
    else
    {
        _impl->op->prepare(_impl->prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries(_impl->aux_mem_req, _impl->workspace);
    }
}
} // namespace arm_compute