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

#include "arm_compute/runtime/GLES_COMPUTE/functions/GCConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;

GCConvolutionLayerReshapeWeights::GCConvolutionLayerReshapeWeights()
    : _weights_reshape_kernel(), _weights_transposed_kernel(), _weights_reshaped(), _transpose1xW(false)
{
}

void GCConvolutionLayerReshapeWeights::configure(const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, bool transpose1xW)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON(is_data_type_quantized_asymmetric(weights->info()->data_type()));
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    const bool       append_biases = (biases != nullptr) && !is_data_type_quantized_asymmetric(weights->info()->data_type());
    const unsigned   bias_element  = (append_biases) ? 1 : 0;
    const IGCTensor *biases_to_use = (append_biases) ? biases : nullptr;

    _transpose1xW = transpose1xW;

    if(transpose1xW)
    {
        // Create tensor to store the reshaped weights
        const unsigned int mat_weights_cols = weights->info()->dimension(3);
        const unsigned int mat_weights_rows = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2) + bias_element;
        TensorShape        shape_wr(mat_weights_cols, mat_weights_rows);
        const DataType     dt                   = weights->info()->data_type();
        const int          fixed_point_position = weights->info()->fixed_point_position();
        TensorInfo         info_wr(shape_wr, 1, dt, fixed_point_position);

        _weights_reshaped.allocator()->init(info_wr);
        _weights_reshape_kernel.configure(weights, biases_to_use, &_weights_reshaped);
        _weights_transposed_kernel.configure(&_weights_reshaped, output);
        _weights_reshaped.allocator()->allocate();
    }
    else
    {
        _weights_reshape_kernel.configure(weights, biases_to_use, output);
    }
}

void GCConvolutionLayerReshapeWeights::run()
{
    GCScheduler::get().dispatch(_weights_reshape_kernel);
    if(_transpose1xW)
    {
        GCScheduler::get().dispatch(_weights_transposed_kernel);
    }
}

GCConvolutionLayer::GCConvolutionLayer()
    : _reshape_weights(), _input_im2col_kernel(), _input_interleave_kernel(), _mm_kernel(), _output_col2im_kernel(), _fill_border(), _input_im2col_reshaped(), _input_interleaved_reshaped(),
      _weights_reshaped(), _weights_transposed(), _gemm_output(), _tmp_output(), _append_bias(false), _is_fully_connected_convolution(false), _are_weights_reshaped(false)
{
}

void GCConvolutionLayer::configure_mm(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output, bool is_interleaved_transposed)
{
    _mm_kernel.configure(input, weights, output, 1.f, is_interleaved_transposed);
}

void GCConvolutionLayer::configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON(!weights_info.are_reshaped() && weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON(!weights_info.are_reshaped() && biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    const DataType dt = input->info()->data_type();

    _append_bias          = (biases != nullptr);
    _are_weights_reshaped = weights_info.are_reshaped();

    const unsigned   bias_element  = (_append_bias) ? 1 : 0;
    const IGCTensor *biases_to_use = (_append_bias) ? biases : nullptr;

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    const unsigned int kernel_width  = (_are_weights_reshaped) ? weights_info.kernel_size().first : weights->info()->dimension(0);
    const unsigned int kernel_height = (_are_weights_reshaped) ? weights_info.kernel_size().second : weights->info()->dimension(1);
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), kernel_width, kernel_height,
                                                 conv_info);

    // Check if its a "fully connected" convolution
    _is_fully_connected_convolution = ((conv_w == 1) && (conv_h == 1));
    const bool run_interleaved      = (!_is_fully_connected_convolution);

    unsigned int mat_weights_cols = weights->info()->dimension(3);
    unsigned int mat_weights_rows = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2) + bias_element;

    // Reshape weights if needed
    if(_are_weights_reshaped)
    {
        if(_is_fully_connected_convolution)
        {
            mat_weights_cols = weights->info()->dimension(0);
            mat_weights_rows = weights->info()->dimension(1);
        }
        else
        {
            mat_weights_cols                         = weights_info.num_kernels();
            const unsigned int quarter_reshaped_cols = weights->info()->dimension(0) / 4;
            mat_weights_rows                         = quarter_reshaped_cols + bias_element;
        }
    }
    else
    {
        if(_is_fully_connected_convolution)
        {
            // Create tensor to store the reshaped weights
            int num_elems_read_per_iteration_x = 1;
            if(dt == DataType::F16)
            {
                num_elems_read_per_iteration_x = 2;
            }
            TensorShape shape_wr((ceil_to_multiple(mat_weights_cols, num_elems_read_per_iteration_x)), mat_weights_rows);
            _weights_reshaped.allocator()->init(weights->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_wr));
            _reshape_weights.configure(weights, biases_to_use, &_weights_reshaped, false /* 1xW transpose */);
        }
        else
        {
            // Create tensor to store transposed weights
            const float transpose_width = 16.0f / input->info()->element_size();
            TensorShape shape_wt(mat_weights_rows * static_cast<unsigned int>(transpose_width), static_cast<unsigned int>(std::ceil(mat_weights_cols / transpose_width)));
            _weights_reshaped.allocator()->init(weights->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_wt));
            _reshape_weights.configure(weights, biases_to_use, &_weights_reshaped, true /* 1xW transpose */);
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

    // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
    TensorInfo im2col_reshaped_info(shape_im2col, 1, dt, input->info()->fixed_point_position());
    _input_im2col_reshaped.allocator()->init(im2col_reshaped_info);

    // Create tensor (interleave) to prepare input tensor for GEMM
    if(run_interleaved)
    {
        TensorShape shape_interleaved = shape_im2col;
        shape_interleaved.set(0, shape_interleaved.x() * 4);
        shape_interleaved.set(1, std::ceil(shape_interleaved.y() / 4.f));

        // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
        TensorInfo interleaved_info(shape_interleaved, 1, dt, input->info()->fixed_point_position());
        _input_interleaved_reshaped.allocator()->init(interleaved_info);
    }

    // Create GEMM output tensor
    TensorShape shape_gemm = _input_im2col_reshaped.info()->tensor_shape();
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    const DataType gemm_data_type = dt;

    // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
    TensorInfo info_gemm(shape_gemm, 1, gemm_data_type, input->info()->fixed_point_position());
    _gemm_output.allocator()->init(info_gemm);

    // Configure kernels
    if(dt == DataType::F16)
    {
        BorderSize border_size = BorderSize(conv_info.pad_top(), conv_info.pad_right(), conv_info.pad_bottom(), conv_info.pad_left());
        input->info()->extend_padding(border_size);
        _fill_border.configure(input, border_size, BorderMode::CONSTANT, PixelValue(0)); // for PAD of im2col fp16: consider it as border
    }
    _input_im2col_kernel.configure(input, &_input_im2col_reshaped, Size2D(kernel_width, kernel_height), conv_info, _append_bias);

    // Configure matrix multiply
    if(run_interleaved)
    {
        _input_interleave_kernel.configure(&_input_im2col_reshaped, &_input_interleaved_reshaped);
        configure_mm(&_input_interleaved_reshaped, weights, &_gemm_output);
        _input_interleaved_reshaped.allocator()->allocate();
    }
    else
    {
        configure_mm(&_input_im2col_reshaped, weights, &_gemm_output, false);
    }
    _input_im2col_reshaped.allocator()->allocate();

    // Configure Col2Im
    _output_col2im_kernel.configure(&_gemm_output, output, std::make_pair(conv_w, conv_h));
    _gemm_output.allocator()->allocate();

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(0) != conv_w) || (output->info()->dimension(1) != conv_h), "Output shape does not match the expected one");

    // Allocate intermediate tensor
    if(!_are_weights_reshaped)
    {
        _weights_reshaped.allocator()->allocate();
    }
}

void GCConvolutionLayer::run()
{
    // Run weights reshaping (Runs once for every configure)
    if(!_are_weights_reshaped)
    {
        _are_weights_reshaped = true;
        _reshape_weights.run();
    }

    // Run im2col
    GCScheduler::get().dispatch(_fill_border);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(_input_im2col_kernel);

    if(!_is_fully_connected_convolution)
    {
        GCScheduler::get().memory_barrier();
        // Run interleave4x4
        GCScheduler::get().dispatch(_input_interleave_kernel);
    }

    GCScheduler::get().memory_barrier();
    // Runs matrix multiply on reshaped matrices
    GCScheduler::get().dispatch(_mm_kernel);

    GCScheduler::get().memory_barrier();
    // Reshape output matrix
    GCScheduler::get().dispatch(_output_col2im_kernel, false);
}
