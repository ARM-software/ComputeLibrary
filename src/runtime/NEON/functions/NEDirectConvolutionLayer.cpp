/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEDirectConvolutionLayerKernel.h"
#include "src/core/NEON/kernels/NEDirectConvolutionLayerOutputStageKernel.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NEDirectConvolutionLayer::~NEDirectConvolutionLayer() = default;

NEDirectConvolutionLayer::NEDirectConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _output_stage_kernel(), _conv_kernel(), _input_border_handler(), _activationlayer_function(), _accumulator(), _has_bias(false),
      _is_activationlayer_enabled(false), _dim_split(Window::DimZ), _is_padding_required()
{
}

void NEDirectConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON(input->info()->data_layout() == DataLayout::UNKNOWN);
    _output_stage_kernel  = arm_compute::support::cpp14::make_unique<NEDirectConvolutionLayerOutputStageKernel>();
    _conv_kernel          = arm_compute::support::cpp14::make_unique<NEDirectConvolutionLayerKernel>();
    _input_border_handler = arm_compute::support::cpp14::make_unique<NEFillBorderKernel>();

    // Free accumulator
    if(_accumulator.buffer() != nullptr)
    {
        _accumulator.allocator()->free();
    }

    _dim_split = input->info()->data_layout() == DataLayout::NCHW ? Window::DimZ : Window::DimY;

    // Check if bias should be added in the convolution result
    _has_bias = (bias != nullptr);

    _conv_kernel->configure(input, weights, output, conv_info);
    if(_has_bias)
    {
        _output_stage_kernel->configure(output, bias);
    }
    _is_padding_required = !_conv_kernel->border_size().empty();

    if(_is_padding_required)
    {
        // Add zero padding XY
        _input_border_handler->configure(input, _conv_kernel->border_size(), BorderMode::CONSTANT, PixelValue(static_cast<float>(0.f)));
    }

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }
}

Status NEDirectConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);

    // output might not be initialized since it can be an intermediate tensor of another layer
    DataType   data_type = input->data_type();
    TensorInfo accumulator(output->clone()->set_is_resizable(true).reset_padding().set_data_type(data_type));

    // Validate Convolution kernel
    ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayerKernel::validate(input, weights, &accumulator, conv_info));

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->dimension(0) != weights->dimension(3),
                                        "Biases size and number of input feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->num_dimensions() > 1, "Biases should be one dimensional");
    }

    // Validate bias kernel
    ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayerOutputStageKernel::validate(&accumulator, bias, output));

    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(output, nullptr, act_info));
    }

    return Status{};
}

void NEDirectConvolutionLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_is_padding_required)
    {
        NEScheduler::get().schedule(_input_border_handler.get(), Window::DimZ);
    }
    NEScheduler::get().schedule(_conv_kernel.get(), _dim_split);
    if(_has_bias)
    {
        NEScheduler::get().schedule(_output_stage_kernel.get(), Window::DimY);
    }

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }
}
} // namespace arm_compute
