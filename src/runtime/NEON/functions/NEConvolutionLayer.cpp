/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/DataTypeUtils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/functions/NEFFTConvolutionLayer.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuConv2d.h"
#include "src/cpu/operators/CpuDirectConv2d.h"
#include "src/cpu/operators/CpuGemmConv2d.h"
#include "src/cpu/operators/CpuGemmDirectConv2d.h"
#include "src/cpu/operators/CpuWinogradConv2d.h"

namespace arm_compute
{
using namespace arm_compute::experimental;

struct NEConvolutionLayer::Impl
{
    MemoryGroup                        memory_group{};
    std::shared_ptr<IMemoryManager>    memory_manager{};
    std::unique_ptr<cpu::ICpuOperator> op{nullptr};
    ITensorPack                        run_pack{};
    ITensorPack                        prep_pack{};
    WorkspaceData<Tensor>              workspace{};
    experimental::MemoryRequirements   aux_mem_req{};
    std::unique_ptr<IFunction>         func{nullptr};
};

NEConvolutionLayer::NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) : _impl(std::make_unique<Impl>())
{
    _impl->memory_manager = std::move(memory_manager);
}

NEConvolutionLayer::~NEConvolutionLayer() = default;

void NEConvolutionLayer::configure(ITensor                   *input,
                                   const ITensor             *weights,
                                   const ITensor             *biases,
                                   ITensor                   *output,
                                   const PadStrideInfo       &conv_info,
                                   const WeightsInfo         &weights_info,
                                   const Size2D              &dilation,
                                   const ActivationLayerInfo &act_info,
                                   bool                       enable_fast_math,
                                   unsigned int               num_groups)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayer::validate(
        input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info,
        weights_info, dilation, act_info, enable_fast_math, num_groups));
    ARM_COMPUTE_LOG_PARAMS(input, weights, biases, output, conv_info, weights_info, dilation, act_info,
                           enable_fast_math, num_groups);

    const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math, num_groups);
    switch (cpu::CpuConv2d::get_convolution_method(input->info(), weights->info(), output->info(), conv_info,
                                                   weights_info, dilation, act_info, enable_fast_math))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::GEMM:
        case ConvolutionMethod::GEMM_CONV2D:
        case ConvolutionMethod::DIRECT:
        {
            auto f = std::make_unique<cpu::CpuConv2d>();
            f->configure(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr),
                         output->info(), conv_info, weights_info, dilation, act_info, enable_fast_math, num_groups);
            _impl->op = std::move(f);
            break;
        }
        case ConvolutionMethod::FFT:
        {
            auto f = std::make_unique<NEFFTConvolutionLayer>(_impl->memory_manager);
            f->configure(input, weights, biases, output, conv_info, act_info);
            _impl->func = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    if (_impl->op)
    {
        _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
        _impl->aux_mem_req  = _impl->op->workspace();
        _impl->run_pack     = {{ACL_SRC_0, input}, {ACL_SRC_1, weights}, {ACL_SRC_2, biases}, {ACL_DST, output}};
        _impl->prep_pack    = {{ACL_SRC_1, weights}, {ACL_SRC_2, biases}};
        _impl->workspace =
            manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
    }
}

Status NEConvolutionLayer::validate(const ITensorInfo         *input,
                                    const ITensorInfo         *weights,
                                    const ITensorInfo         *biases,
                                    const ITensorInfo         *output,
                                    const PadStrideInfo       &conv_info,
                                    const WeightsInfo         &weights_info,
                                    const Size2D              &dilation,
                                    const ActivationLayerInfo &act_info,
                                    bool                       enable_fast_math,
                                    unsigned int               num_groups)
{
    const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math, num_groups);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!weights->are_values_constant(), "Dynamic weights are not supported");

    // Biases with dynamic values are not supported with quantized inputs.
    if (biases)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((!biases->are_values_constant() && is_data_type_quantized(input->data_type())),
                                        "Dynamic Biases are not supported with quantized input data.");
    }

    switch (cpu::CpuConv2d::get_convolution_method(input, weights, output, conv_info, weights_info, dilation, act_info,
                                                   enable_fast_math))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::GEMM:
        case ConvolutionMethod::GEMM_CONV2D:
        case ConvolutionMethod::DIRECT:
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuConv2d::validate(input, weights, biases, output, conv_info,
                                                                 weights_info, dilation, act_info, enable_fast_math,
                                                                 num_groups));
            break;
        case ConvolutionMethod::FFT:
            ARM_COMPUTE_RETURN_ON_ERROR(
                NEFFTConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info));
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
    return Status{};
}

ConvolutionMethod NEConvolutionLayer::get_convolution_method(const ITensorInfo         *input,
                                                             const ITensorInfo         *weights,
                                                             const ITensorInfo         *output,
                                                             const PadStrideInfo       &conv_info,
                                                             const WeightsInfo         &weights_info,
                                                             const Size2D              &dilation,
                                                             const ActivationLayerInfo &act_info,
                                                             bool                       enable_fast_math)
{
    return cpu::CpuConv2d::get_convolution_method(input, weights, output, conv_info, weights_info, dilation, act_info,
                                                  enable_fast_math);
}

void NEConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    if (_impl->func)
    {
        _impl->func->run();
    }
    else
    {
        _impl->op->run(_impl->run_pack);
    }
}

void NEConvolutionLayer::prepare()
{
    if (_impl->func)
    {
        _impl->func->prepare();
    }
    else
    {
        _impl->op->prepare(_impl->prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(_impl->aux_mem_req, _impl->workspace);
    }
}
} // namespace arm_compute
