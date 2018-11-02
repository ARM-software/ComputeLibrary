/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <tuple>
#include <utility>

namespace arm_compute
{
NEConvolutionLayer::NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) //NOLINT
    : _memory_manager(std::move(memory_manager)),
      _function()
{
}

void NEConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                   const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayer::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info, weights_info, dilation, act_info,
                                                            enable_fast_math));

    switch(NEConvolutionLayer::get_convolution_method(input->info(), weights->info(), output->info(), conv_info, weights_info, dilation, act_info))
    {
        case ConvolutionMethod::WINOGRAD:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEWinogradConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, act_info, enable_fast_math);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEGEMMConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, weights_info, dilation, act_info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::DIRECT:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEDirectConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, act_info);
            _function = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
}

Status NEConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                    const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    switch(NEConvolutionLayer::get_convolution_method(input, weights, output, conv_info, weights_info, dilation, act_info))
    {
        case ConvolutionMethod::WINOGRAD:
            //Validate Winograd
            ARM_COMPUTE_RETURN_ON_ERROR(NEWinogradConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info, enable_fast_math));
            break;
        case ConvolutionMethod::GEMM:
            //Validate Gemm-based Convolution
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMConvolutionLayer::validate(input, weights, biases, output, conv_info, weights_info, dilation, act_info));
            break;
        case ConvolutionMethod::DIRECT:
            //Validate Gemm-based Convolution
            ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info));
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

ConvolutionMethod NEConvolutionLayer::get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights,
                                                             const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                             const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, weights);
    ARM_COMPUTE_UNUSED(weights_info);

    if(dilation != Size2D(1U, 1U) || Scheduler::get().cpu_info().get_cpu_model() == CPUModel::A53
       || input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL)) <= 16)
    {
        return ConvolutionMethod::GEMM;
    }

    return bool(NEWinogradConvolutionLayer::validate(input, weights, nullptr, output, conv_info, act_info, enable_fast_math)) ? ConvolutionMethod::WINOGRAD : ConvolutionMethod::GEMM;
}

void NEConvolutionLayer::run()
{
    _function->run();
}
} // namespace arm_compute
