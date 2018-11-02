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
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLConvolutionLayer::CLConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _function()
{
}

void CLConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLConvolutionLayer::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info, weights_info));

    switch(CLConvolutionLayer::get_convolution_method(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info,
                                                      weights_info, CLScheduler::get().target()))
    {
        case ConvolutionMethod::DIRECT:
        {
            auto f = arm_compute::support::cpp14::make_unique<CLDirectConvolutionLayer>();
            f->configure(input, weights, biases, output, conv_info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            auto f = arm_compute::support::cpp14::make_unique<CLGEMMConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, weights_info);
            _function = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
}

Status CLConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                    const WeightsInfo &weights_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);

    //Configure if the parameters match the direct convolution or the gemm-based
    const GPUTarget gpu_target = CLScheduler::get().target();

    switch(CLConvolutionLayer::get_convolution_method(input, weights, biases, output, conv_info, weights_info, gpu_target))
    {
        case ConvolutionMethod::DIRECT:
        {
            // Validate direct convolution layer
            CLDirectConvolutionLayer::validate(input, weights, biases, output, conv_info);
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            // Validate gemm-based convolution layer
            CLGEMMConvolutionLayer::validate(input, weights, biases, output, conv_info, weights_info);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

ConvolutionMethod CLConvolutionLayer::get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                             const WeightsInfo &weights_info, const GPUTarget gpu_target)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(weights);
    ARM_COMPUTE_UNUSED(biases);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(conv_info);
    ARM_COMPUTE_UNUSED(weights_info);
    ARM_COMPUTE_UNUSED(gpu_target);

    return ConvolutionMethod::GEMM;
}

void CLConvolutionLayer::run()
{
    _function->run();
}
