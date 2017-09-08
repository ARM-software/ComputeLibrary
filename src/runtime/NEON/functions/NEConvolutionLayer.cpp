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
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

NEConvolutionLayerReshapeWeights::NEConvolutionLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _weights_reshape_kernel(), _weights_transposed_kernel(), _weights_reshaped(), _transpose1xW(false)
{
}

void NEConvolutionLayerReshapeWeights::configure(const ITensor *weights, const ITensor *biases, ITensor *output, bool transpose1xW)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(weights, output);
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(weights, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    // Check if bias are present, if yes they will be embedded to the weights matrix
    const bool _has_bias = (biases != nullptr);

    _transpose1xW = transpose1xW;

    if(transpose1xW)
    {
        // Create tensor to store the reshaped weights
        const unsigned int mat_weights_cols = weights->info()->dimension(3);
        const unsigned int mat_weights_rows = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2) + (_has_bias ? 1 : 0);
        TensorShape        shape_wr(mat_weights_cols, mat_weights_rows);
        TensorInfo         info_wr(shape_wr, 1, weights->info()->data_type(), weights->info()->fixed_point_position());

        _weights_reshaped.allocator()->init(info_wr);
        _memory_group.manage(&_weights_reshaped);
        _weights_reshape_kernel.configure(weights, biases, &_weights_reshaped);
        _weights_transposed_kernel.configure(&_weights_reshaped, output);
        _weights_reshaped.allocator()->allocate();
    }
    else
    {
        _weights_reshape_kernel.configure(weights, biases, output);
    }
}

void NEConvolutionLayerReshapeWeights::run()
{
    _memory_group.acquire();

    NEScheduler::get().schedule(&_weights_reshape_kernel, 3);
    if(_transpose1xW)
    {
        NEScheduler::get().schedule(&_weights_transposed_kernel, Window::DimY);
    }

    _memory_group.release();
}

NEConvolutionLayer::NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _input_im2col_kernel(), _input_interleave_kernel(), _reshape_weights(), _mm_kernel(), _output_col2im_kernel(), _input_im2col_reshaped(),
      _input_interleaved_reshaped(), _weights_reshaped(), _gemm_output(), _has_bias(false), _is_fully_connected_convolution(false), _are_weights_reshaped(false)
{
}

void NEConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, weights);
    ARM_COMPUTE_ERROR_ON(!weights_info.are_reshaped() && weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, biases);
        ARM_COMPUTE_ERROR_ON(!weights_info.are_reshaped() && biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    const DataType dt                   = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();

    _has_bias             = (biases != nullptr);
    _are_weights_reshaped = weights_info.are_reshaped();

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    unsigned int pad_x    = 0;
    unsigned int pad_y    = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();
    std::tie(pad_x, pad_y)       = conv_info.pad();

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    const unsigned int kernel_width  = (_are_weights_reshaped) ? weights_info.kernel_size().first : weights->info()->dimension(0);
    const unsigned int kernel_height = (_are_weights_reshaped) ? weights_info.kernel_size().second : weights->info()->dimension(1);
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), kernel_width, kernel_height,
                                                 conv_info);

    // Check if its a "fully connected" convolution
    _is_fully_connected_convolution = ((conv_w == 1) && (conv_h == 1));

    unsigned int mat_weights_cols = weights->info()->dimension(3);
    unsigned int mat_weights_rows = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2) + (_has_bias ? 1 : 0);

    // Reshape weights if needed
    if(_are_weights_reshaped)
    {
        mat_weights_cols                         = weights_info.num_kernels();
        const unsigned int quarter_reshaped_cols = weights->info()->dimension(0) / 4;
        mat_weights_rows                         = (_has_bias ? 1 + quarter_reshaped_cols : quarter_reshaped_cols);
    }
    else
    {
        if(_is_fully_connected_convolution)
        {
            // Create tensor to store the reshaped weights
            TensorShape shape_wr(mat_weights_cols, mat_weights_rows);
            TensorInfo  info_wr(shape_wr, 1, dt, fixed_point_position);
            _weights_reshaped.allocator()->init(info_wr);
            _reshape_weights.configure(weights, biases, &_weights_reshaped, false /* 1xW transpose */);
        }
        else
        {
            // Create tensor to store transposed weights
            const float transpose_width = 16.0f / input->info()->element_size();
            TensorShape shape_wt(mat_weights_rows * static_cast<unsigned int>(transpose_width), static_cast<unsigned int>(std::ceil(mat_weights_cols / transpose_width)));
            TensorInfo  info_wt(shape_wt, 1, dt, fixed_point_position);
            _weights_reshaped.allocator()->init(info_wt);
            _reshape_weights.configure(weights, biases, &_weights_reshaped, true /* 1xW transpose */);
        }
        weights = &_weights_reshaped;
    }

    // Create tensor to store im2col reshaped inputs
    const unsigned int mat_input_cols = mat_weights_rows;
    const unsigned int mat_input_rows = conv_w * conv_h;
    TensorShape        shape_im2col   = input->info()->tensor_shape();
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);
    shape_im2col.set(2, 1);
    _input_im2col_reshaped.allocator()->init(TensorInfo(shape_im2col, 1, dt, fixed_point_position));
    _memory_group.manage(&_input_im2col_reshaped);

    // Create tensor (interleave) to prepare input tensor for GEMM
    if(!_is_fully_connected_convolution)
    {
        TensorShape shape_interleaved = shape_im2col;
        shape_interleaved.set(0, shape_interleaved.x() * 4);
        shape_interleaved.set(1, std::ceil(shape_interleaved.y() / 4.f));
        _input_interleaved_reshaped.allocator()->init(TensorInfo(shape_interleaved, 1, dt, fixed_point_position));
        _memory_group.manage(&_input_interleaved_reshaped);
    }

    // Create GEMM output tensor
    TensorShape shape_gemm = _input_im2col_reshaped.info()->tensor_shape();
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    _gemm_output.allocator()->init(TensorInfo(shape_gemm, 1, dt, fixed_point_position));
    _memory_group.manage(&_gemm_output);

    // Configure kernels
    _input_im2col_kernel.configure(input, &_input_im2col_reshaped, Size2D(kernel_width, kernel_height), conv_info, _has_bias);
    if(_is_fully_connected_convolution)
    {
        _mm_kernel.configure(&_input_im2col_reshaped, weights, &_gemm_output, 1.0f);
    }
    else
    {
        _input_interleave_kernel.configure(&_input_im2col_reshaped, &_input_interleaved_reshaped);
        _mm_kernel.configure(&_input_interleaved_reshaped, weights, &_gemm_output, 1.0f);
        _input_interleaved_reshaped.allocator()->allocate();
    }
    _input_im2col_reshaped.allocator()->allocate();
    _output_col2im_kernel.configure(&_gemm_output, output, std::make_pair(conv_w, conv_h));
    _gemm_output.allocator()->allocate();

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(0) != conv_w) || (output->info()->dimension(1) != conv_h), "Output shape does not match the expected one");

    // Allocate intermediate tensor
    if(!_are_weights_reshaped)
    {
        _weights_reshaped.allocator()->allocate();
    }
}

void NEConvolutionLayer::run()
{
    // Run weights reshaping (Runs once for every configure)
    if(!_are_weights_reshaped)
    {
        _are_weights_reshaped = true;
        _reshape_weights.run();
    }

    _memory_group.acquire();

    // Run input reshaping
    NEScheduler::get().schedule(&_input_im2col_kernel, Window::DimY);
    if(!_is_fully_connected_convolution)
    {
        // Run interleave
        NEScheduler::get().schedule(&_input_interleave_kernel, Window::DimY);
    }

    // Runs matrix multiply on reshaped matrices
    NEScheduler::get().schedule(&_mm_kernel, Window::DimY);

    // Reshape output matrix
    NEScheduler::get().schedule(&_output_col2im_kernel, Window::DimY);

    _memory_group.release();
}
