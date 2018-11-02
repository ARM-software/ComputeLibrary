/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLDepthwiseConvolution.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLDepthwiseConvolution3x3::CLDepthwiseConvolution3x3()
    : _kernel(), _border_handler()
{
}

void CLDepthwiseConvolution3x3::configure(ICLTensor *input, ICLTensor *output, const ICLTensor *weights, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    _kernel.configure(input, output, weights, conv_info);
    _border_handler.configure(input, _kernel.border_size(), BorderMode::CONSTANT, PixelValue(0));
}

void CLDepthwiseConvolution3x3::run()
{
    CLScheduler::get().enqueue(_border_handler);
    CLScheduler::get().enqueue(_kernel);
}

CLDepthwiseConvolution::CLDepthwiseConvolution()
    : _im2col_kernel(), _weights_reshape_kernel(), _v2mm_kernel(), _vector_to_tensor_kernel(), _v2mm_input_fill_border(), _v2mm_weights_fill_border(), _input_reshaped(), _weights_reshaped(),
      _v2mm_output()
{
}

void CLDepthwiseConvolution::configure(ICLTensor *input, ICLTensor *output, const ICLTensor *weights, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != weights->info()->dimension(2));

    const size_t weights_w = weights->info()->dimension(0);
    const size_t weights_h = weights->info()->dimension(1);
    const size_t weights_z = weights->info()->dimension(2);

    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), weights_w, weights_h, conv_info);

    // Set up intermediate tensors
    const size_t patch_size = weights_w * weights_h;
    const size_t conv_size  = conv_w * conv_h;

    TensorShape shape_im2col = input->info()->tensor_shape();
    shape_im2col.set(0, patch_size);
    shape_im2col.set(1, conv_size);
    shape_im2col.set(2, weights_z);

    const TensorShape shape_weights_reshape(patch_size, weights_z);
    TensorShape       shape_v2mm_out = output->info()->tensor_shape();
    shape_v2mm_out.set(0, conv_size * weights_z);
    shape_v2mm_out.set(1, 1);
    shape_v2mm_out.set(2, 1);

    const TensorInfo info_im2col(shape_im2col, 1, input->info()->data_type(), input->info()->fixed_point_position());
    const TensorInfo info_weights_reshape(shape_weights_reshape, 1, weights->info()->data_type(), weights->info()->fixed_point_position());
    const TensorInfo info_v2mm_out(shape_v2mm_out, 1, input->info()->data_type(), input->info()->fixed_point_position());

    _input_reshaped.allocator()->init(info_im2col);
    _weights_reshaped.allocator()->init(info_weights_reshape);
    _v2mm_output.allocator()->init(info_v2mm_out);

    // Configure kernels
    _im2col_kernel.configure(input, &_input_reshaped, Size2D(weights_w, weights_h), conv_info);
    _weights_reshape_kernel.configure(weights, &_weights_reshaped);
    _v2mm_kernel.configure(&_input_reshaped, &_weights_reshaped, &_v2mm_output);
    _vector_to_tensor_kernel.configure(&_v2mm_output, output, conv_w, conv_h);

    BorderSize border_size = _v2mm_kernel.border_size();
    _v2mm_input_fill_border.configure(&_input_reshaped, border_size, BorderMode::CONSTANT, PixelValue(0));

    border_size.bottom = 0;
    _v2mm_weights_fill_border.configure(&_weights_reshaped, border_size, BorderMode::CONSTANT, PixelValue(0));

    // Allocate intermediate tensors
    _input_reshaped.allocator()->allocate();
    _weights_reshaped.allocator()->allocate();
    _v2mm_output.allocator()->allocate();
}

void CLDepthwiseConvolution::run()
{
    CLScheduler::get().enqueue(_im2col_kernel);

    CLScheduler::get().enqueue(_weights_reshape_kernel);

    CLScheduler::get().enqueue(_v2mm_input_fill_border);
    CLScheduler::get().enqueue(_v2mm_weights_fill_border);
    CLScheduler::get().enqueue(_v2mm_kernel);

    CLScheduler::get().enqueue(_vector_to_tensor_kernel);
}
