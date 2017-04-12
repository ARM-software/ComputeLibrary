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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

NEConvolutionLayer::NEConvolutionLayer()
    : _input_im2col_kernel(), _input_interleave_kernel(), _weights_reshape_kernel(), _weights_transposed_kernel(), _mm_kernel(), _output_col2im_kernel(), _input_im2col_reshaped(),
      _input_interleaved_reshaped(), _weights_reshaped(), _weights_transposed(), _gemm_output(), _is_first_run(false), _has_bias(false)
{
}

void NEConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F32);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    _has_bias     = (biases != nullptr);
    _is_first_run = true;

    // Get parameters for conv_info
    unsigned int stride_x, stride_y, pad_x, pad_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();
    std::tie(pad_x, pad_y)       = conv_info.pad();

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), weights->info()->dimension(0),
                                                 stride_x, stride_y, pad_x, pad_y, conv_info.round());
    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(0) != conv_w) || (output->info()->dimension(1) != conv_h), "Output shape does not match the expected one");

    // Create tensor to store the reshaped weights
    const size_t      mat_weights_cols = weights->info()->dimension(3);
    const size_t      mat_weights_rows = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2) + ((_has_bias) ? 1 : 0);
    const TensorShape shape_wr(mat_weights_cols, mat_weights_rows);
    TensorInfo        info_wr(shape_wr, 1, weights->info()->data_type());
    _weights_reshaped.allocator()->init(info_wr);

    // Create tensor to store transposed weights
    TensorShape shape_wt(mat_weights_rows * 4, static_cast<size_t>(std::ceil(mat_weights_cols / 4.f)));
    TensorInfo  info_wt(shape_wt, 1, weights->info()->data_type());
    _weights_transposed.allocator()->init(info_wt);

    // Create tensor to store im2col reshaped inputs
    const size_t mat_input_cols = mat_weights_rows;
    const size_t mat_input_rows = conv_w * conv_h;
    TensorShape  shape_im2col   = input->info()->tensor_shape();
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);
    shape_im2col.set(2, 1);
    TensorInfo info_im2col(shape_im2col, 1, input->info()->data_type());
    _input_im2col_reshaped.allocator()->init(info_im2col);

    // Create tensor to prepare input tensor for GEMM
    TensorShape shape_interleaved = shape_im2col;
    shape_interleaved.set(0, shape_interleaved.x() * 4);
    shape_interleaved.set(1, std::ceil(static_cast<float>(shape_interleaved.y()) / 4));
    TensorInfo info_interleaved(shape_interleaved, 1, input->info()->data_type());
    _input_interleaved_reshaped.allocator()->init(info_interleaved);

    // Create GEMM output tensor
    TensorShape shape_gemm = _input_im2col_reshaped.info()->tensor_shape();
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    TensorInfo info_gemm(shape_gemm, 1, input->info()->data_type());
    _gemm_output.allocator()->init(info_gemm);

    // Configure kernels
    _input_im2col_kernel.configure(input, &_input_im2col_reshaped, std::make_pair(conv_w, conv_h), conv_info, _has_bias);
    _input_interleave_kernel.configure(&_input_im2col_reshaped, &_input_interleaved_reshaped);
    _weights_reshape_kernel.configure(weights, biases, &_weights_reshaped);
    _weights_transposed_kernel.configure(&_weights_reshaped, &_weights_transposed);
    _mm_kernel.configure(&_input_interleaved_reshaped, &_weights_transposed, &_gemm_output, 1.0f);
    _output_col2im_kernel.configure(&_gemm_output, output, std::make_pair(conv_w, conv_h));

    // Allocate the tensors once the all configure methods have been called
    _weights_reshaped.allocator()->allocate();
    _weights_transposed.allocator()->allocate();
    _input_im2col_reshaped.allocator()->allocate();
    _input_interleaved_reshaped.allocator()->allocate();
    _gemm_output.allocator()->allocate();
}

void NEConvolutionLayer::run()
{
    // Run weights reshaping (Runs once for every configure)
    if(_is_first_run)
    {
        _is_first_run = false;
        NEScheduler::get().multithread(&_weights_reshape_kernel, 3);
        NEScheduler::get().multithread(&_weights_transposed_kernel);
    }

    // Run input reshaping
    NEScheduler::get().multithread(&_input_im2col_kernel);

    // Run interleave
    NEScheduler::get().multithread(&_input_interleave_kernel);

    // Runs GEMM on reshaped matrices
    NEScheduler::get().multithread(&_mm_kernel);

    // Reshape output matrix
    NEScheduler::get().multithread(&_output_col2im_kernel);
}
