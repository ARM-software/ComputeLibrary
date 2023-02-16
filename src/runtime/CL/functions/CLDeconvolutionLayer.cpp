/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayer.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/IClOperator.h"
#include "src/gpu/cl/operators/ClTransposedConvolution.h"

#include "src/common/utils/Log.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

struct CLDeconvolutionLayer::Impl
{
    const ICLTensor                     *src{ nullptr };
    const ICLTensor                     *weights{ nullptr };
    const ICLTensor                     *biases{ nullptr };
    ICLTensor                           *dst{ nullptr };
    std::unique_ptr<opencl::IClOperator> op{ nullptr };
};

CLDeconvolutionLayer::~CLDeconvolutionLayer() = default;

CLDeconvolutionLayer::CLDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _function(), _impl(std::make_unique<Impl>())
{
}

void CLDeconvolutionLayer::configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &deconv_info,
                                     const WeightsInfo &weights_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, bias, output, deconv_info, weights_info);
}

void CLDeconvolutionLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &deconv_info,
                                     const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_LOG_PARAMS(input, weights, bias, output, deconv_info, weights_info);

    switch(CLDeconvolutionLayer::get_deconvolution_method(input->info(), weights->info(), nullptr, output->info(), deconv_info, weights_info))
    {
        case DeconvolutionMethod::DIRECT:
        {
            auto op = std::make_unique<opencl::ClTransposedConvolution>();
            op->configure(compile_context, input->info(), weights->info(), bias != nullptr ? bias->info() : nullptr, output->info(), deconv_info);

            _impl->src     = input;
            _impl->weights = weights;
            _impl->biases  = bias;
            _impl->dst     = output;

            _impl->op = std::move(op);
            break;
        }
        case DeconvolutionMethod::UPSCALE_CONV2D:
        {
            auto f = std::make_unique<CLDirectDeconvolutionLayer>();
            f->configure(compile_context, input, weights, bias, output, deconv_info, weights_info);
            _function = std::move(f);
            break;
        }
        case DeconvolutionMethod::GEMM:
        {
            auto f = std::make_unique<CLGEMMDeconvolutionLayer>(_memory_manager);
            f->configure(compile_context, input, weights, bias, output, deconv_info);
            _function = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
}

Status CLDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &deconv_info,
                                      const WeightsInfo &weights_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    switch(CLDeconvolutionLayer::get_deconvolution_method(input, weights, bias, output, deconv_info, weights_info))
    {
        case DeconvolutionMethod::DIRECT:
        {
            // Validate transposed convolution operator
            ARM_COMPUTE_RETURN_ON_ERROR(opencl::ClTransposedConvolution::validate(input, weights, bias, output, deconv_info));
            break;
        }
        case DeconvolutionMethod::UPSCALE_CONV2D:
        {
            // Validate direct convolution layer
            ARM_COMPUTE_RETURN_ON_ERROR(CLDirectDeconvolutionLayer::validate(input, weights, bias, output, deconv_info, weights_info));
            break;
        }
        case DeconvolutionMethod::GEMM:
        {
            // Validate gemm-based convolution layer
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMDeconvolutionLayer::validate(input, weights, bias, output, deconv_info));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

DeconvolutionMethod CLDeconvolutionLayer::get_deconvolution_method(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &deconv_info,
                                                                   const WeightsInfo &weights_info)
{
    ARM_COMPUTE_UNUSED(output, bias, weights_info);

    if(is_data_type_quantized_per_channel(weights->data_type()))
    {
        return DeconvolutionMethod::UPSCALE_CONV2D;
    }

    const DataLayout data_layout = input->data_layout();

    const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t idx_n = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);
    const size_t ofm   = weights->tensor_shape()[idx_n];

    if(weights->dimension(idx_w) != deconv_info.stride().first || weights->dimension(idx_h) != deconv_info.stride().second)
    {
        if(input->data_layout() == DataLayout::NHWC && ofm <= 16)
        {
            return DeconvolutionMethod::DIRECT;
        }
        else
        {
            return DeconvolutionMethod::UPSCALE_CONV2D;
        }
    }

    return DeconvolutionMethod::GEMM;
}

void CLDeconvolutionLayer::run()
{
    prepare();

    if(_impl->op != nullptr)
    {
        // Optimized Operator will be used
        ITensorPack pack;

        pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
        pack.add_tensor(TensorType::ACL_SRC_1, _impl->weights);
        pack.add_tensor(TensorType::ACL_SRC_2, _impl->biases);
        pack.add_tensor(TensorType::ACL_DST, _impl->dst);

        _impl->op->run(pack);
    }
    else
    {
        _function->run();
    }
}

void CLDeconvolutionLayer::prepare()
{
    if(_impl->op == nullptr)
    {
        _function->prepare();
    }
}
