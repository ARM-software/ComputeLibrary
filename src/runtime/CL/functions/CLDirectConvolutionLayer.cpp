/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLDirectConvolutionLayerKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLDirectConvolutionLayer::CLDirectConvolutionLayer()
    : _direct_conv_kernel(), _input_border_handler(), _activationlayer_function(), _is_activationlayer_enabled(false)
{
}

void CLDirectConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    // Set GPU target
    _direct_conv_kernel.set_target(CLScheduler::get().target());

    // Configure direct convolution
    _direct_conv_kernel.configure(input, weights, biases, output, conv_info);

    // Configure border handler
    PixelValue &&zero_value(0.f);
    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        zero_value = PixelValue(static_cast<uint8_t>(input->info()->quantization_info().uniform().offset));
    }
    _input_border_handler.configure(input, _direct_conv_kernel.border_size(), BorderMode::CONSTANT, zero_value);

    // Tune kernels
    CLScheduler::get().tune_kernel_static(_direct_conv_kernel);

    _is_activationlayer_enabled = act_info.enabled();

    //Configure Activation Layer
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }
}

Status CLDirectConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(CLDirectConvolutionLayerKernel::validate(input, weights, biases, output, conv_info, CLScheduler::get().target()));
    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(output, nullptr, act_info));
    }
    return Status{};
}

void CLDirectConvolutionLayer::run()
{
    // Run border handler
    CLScheduler::get().enqueue(_input_border_handler, false);

    // Run direct convolution
    CLScheduler::get().enqueue(_direct_conv_kernel);

    //Run Activation Layer
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }
}
