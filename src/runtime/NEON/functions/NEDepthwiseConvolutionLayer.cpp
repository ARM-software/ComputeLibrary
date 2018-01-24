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
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEDepthwiseConvolutionLayer3x3::NEDepthwiseConvolutionLayer3x3()
    : _kernel(), _output_stage_kernel(), _border_handler(), _accumulator(), _has_bias(false), _is_quantized(false)
{
}

void NEDepthwiseConvolutionLayer3x3::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    PixelValue zero_value(0.f);

    _is_quantized = is_data_type_quantized_asymmetric(input->info()->data_type());
    _has_bias     = biases != nullptr;

    // Allocate the intermediate accumulator tensor in case of fixed point input
    if(_is_quantized)
    {
        _accumulator.allocator()->init(TensorInfo(output->info()->tensor_shape(), 1, DataType::S32));
        _accumulator.info()->set_quantization_info(input->info()->quantization_info());
        zero_value = PixelValue(static_cast<uint32_t>(input->info()->quantization_info().offset));
    }

    // Configure depthwise convolution kernel
    _kernel.configure(input, weights, (_is_quantized) ? &_accumulator : output, conv_info);

    // Configure border handler
    _border_handler.configure(input, _kernel.border_size(), BorderMode::CONSTANT, zero_value);

    // Configure biases accumulation
    if(_has_bias || _is_quantized)
    {
        if(_is_quantized)
        {
            float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output->info()->quantization_info().scale;
            int   output_multiplier, output_shift;
            quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
            _output_stage_kernel.configure(&_accumulator, biases, output, output_multiplier, output_shift, output->info()->quantization_info().offset);
            _accumulator.allocator()->allocate();
        }
        else
        {
            _output_stage_kernel.configure(output, biases);
        }
    }
}

void NEDepthwiseConvolutionLayer3x3::run()
{
    NEScheduler::get().schedule(&_border_handler, Window::DimX);
    NEScheduler::get().schedule(&_kernel, Window::DimX);
    if(_has_bias || _is_quantized)
    {
        NEScheduler::get().schedule(&_output_stage_kernel, Window::DimX);
    }
}

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayer()
    : _im2col_kernel(), _weights_reshape_kernel(), _v2mm_kernel(), _vector_to_tensor_kernel(), _input_reshaped(), _weights_reshaped(), _v2mm_output()
{
}

void NEDepthwiseConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != weights->info()->dimension(2));

    const size_t weights_w = weights->info()->dimension(0);
    const size_t weights_h = weights->info()->dimension(1);
    const size_t weights_z = weights->info()->dimension(2);

    bool has_bias = (biases != nullptr);

    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), weights_w, weights_h, conv_info);

    // Set up intermediate tensors
    const size_t patch_size = weights_w * weights_h + ((has_bias) ? 1 : 0);
    const size_t conv_size  = conv_w * conv_h;

    // Im2Col configuration
    TensorShape shape_im2col = input->info()->tensor_shape();
    shape_im2col.set(0, patch_size);
    shape_im2col.set(1, conv_size);
    shape_im2col.set(2, weights_z);
    const TensorInfo info_im2col(shape_im2col, 1, input->info()->data_type(), input->info()->fixed_point_position());
    _input_reshaped.allocator()->init(info_im2col);
    _im2col_kernel.configure(input, &_input_reshaped, Size2D(weights_w, weights_h), conv_info, has_bias);

    // Weights reshape configuration
    const TensorShape shape_weights_reshape(patch_size, weights_z);
    const TensorInfo  info_weights_reshape(shape_weights_reshape, 1, weights->info()->data_type(), weights->info()->fixed_point_position());
    _weights_reshaped.allocator()->init(info_weights_reshape);
    _weights_reshape_kernel.configure(weights, &_weights_reshaped, biases);

    // GEMV configuration
    TensorShape shape_v2mm_out = input->info()->tensor_shape();
    shape_v2mm_out.set(0, conv_size * weights_z);
    shape_v2mm_out.set(1, 1);
    shape_v2mm_out.set(2, 1);
    const TensorInfo info_v2mm_out(shape_v2mm_out, 1, input->info()->data_type(), input->info()->fixed_point_position());
    _v2mm_output.allocator()->init(info_v2mm_out);
    _v2mm_kernel.configure(&_input_reshaped, &_weights_reshaped, &_v2mm_output);
    _vector_to_tensor_kernel.configure(&_v2mm_output, output, conv_w, conv_h);

    // Allocate intermediate tensors
    _input_reshaped.allocator()->allocate();
    _weights_reshaped.allocator()->allocate();
    _v2mm_output.allocator()->allocate();
}

void NEDepthwiseConvolutionLayer::run()
{
    NEScheduler::get().schedule(&_im2col_kernel, Window::DimX);
    NEScheduler::get().schedule(&_weights_reshape_kernel, Window::DimX);
    NEScheduler::get().schedule(&_v2mm_kernel, Window::DimX);
    NEScheduler::get().schedule(&_vector_to_tensor_kernel, Window::DimX);
}