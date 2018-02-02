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

namespace arm_compute
{
NEConvolutionLayer::NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _function()
{
}

void NEConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayer::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info, weights_info));

    switch(NEConvolutionLayer::get_convolution_method(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info,
                                                      weights_info))
    {
        case ConvolutionMethod::WINOGRAD:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEWinogradLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEGEMMConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, weights_info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::DIRECT:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEDirectConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info);
            _function = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
}

Status NEConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                    const WeightsInfo &weights_info)
{
    switch(NEConvolutionLayer::get_convolution_method(input, weights, biases, output, conv_info, weights_info))
    {
        case ConvolutionMethod::WINOGRAD:
            //Validate Winograd
            NEWinogradLayer::validate(input, weights, biases, output, conv_info);
            break;
        case ConvolutionMethod::GEMM:
            //Validate Gemm-based Convolution
            NEGEMMConvolutionLayer::validate(input, weights, biases, output, conv_info, weights_info);
            break;
        case ConvolutionMethod::DIRECT:
            //Validate Gemm-based Convolution
            NEDirectConvolutionLayer::validate(input, weights, biases, output, conv_info);
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

ConvolutionMethod NEConvolutionLayer::get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                             const WeightsInfo &weights_info)
{
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(weights_info);
    if((input->data_type() == DataType::F32) && (weights->dimension(0) == 3) && (weights->dimension(1) == 3) && (weights->num_dimensions() <= 4) && (conv_info.stride().first == 1)
       && (conv_info.stride().second == 1) && (biases != nullptr))
    {
        return ConvolutionMethod::WINOGRAD;
    }
    return ConvolutionMethod::GEMM;
}

void NEConvolutionLayer::run()
{
    _function->run();
}
} // namespace arm_compute
