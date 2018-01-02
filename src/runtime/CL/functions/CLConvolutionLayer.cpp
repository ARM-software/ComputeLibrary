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
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;

CLConvolutionLayerReshapeWeights::CLConvolutionLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _weights_reshape_kernel(), _weights_transposed_kernel(), _weights_reshaped(), _transpose1xW(false)
{
}

void CLConvolutionLayerReshapeWeights::configure(const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, bool transpose1xW)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(weights, output);
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
    const ICLTensor *biases_to_use = (append_biases) ? biases : nullptr;

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
        _memory_group.manage(&_weights_reshaped);
        _weights_reshape_kernel.configure(weights, biases_to_use, &_weights_reshaped);
        _weights_transposed_kernel.configure(&_weights_reshaped, output);
        _weights_reshaped.allocator()->allocate();
    }
    else
    {
        _weights_reshape_kernel.configure(weights, biases_to_use, output);
    }
}

void CLConvolutionLayerReshapeWeights::run()
{
    _memory_group.acquire();

    cl::CommandQueue q = CLScheduler::get().queue();
    CLScheduler::get().enqueue(_weights_reshape_kernel);
    if(_transpose1xW)
    {
        CLScheduler::get().enqueue(_weights_transposed_kernel);
    }

    _memory_group.release();
}

CLConvolutionLayer::CLConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _reshape_weights(), _input_im2col_kernel(), _input_interleave_kernel(), _mm_kernel(), _mm_gemmlowp(memory_manager), _gemmlowp_output_stage(), _output_col2im_kernel(),
      _input_im2col_reshaped(), _input_interleaved_reshaped(), _weights_reshaped(), _weights_transposed(), _gemm_output(), _tmp_output(), _append_bias(false), _is_fully_connected_convolution(false),
      _are_weights_reshaped(false), _is_quantized(false)
{
}

void CLConvolutionLayer::configure_mm(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output, bool is_interleaved_transposed)
{
    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.scale, -input_quantization_info.offset));
        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.scale, -weights_quantization_info.offset));

        _mm_gemmlowp.configure(input, weights, output, GEMMInfo(false, false, true /* Reshape weights only for the first run*/));

        // Revert back QuantizatioInfo as input and weights could be used in other convolution layers
        input->info()->set_quantization_info(input_quantization_info);
        weights->info()->set_quantization_info(weights_quantization_info);
    }
    else
    {
        _mm_kernel.configure(input, weights, output, 1.f, is_interleaved_transposed);
    }
}

void CLConvolutionLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, weights);
    ARM_COMPUTE_ERROR_ON(!weights_info.are_reshaped() && weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(weights_info.are_reshaped() && is_data_type_quantized_asymmetric(input->info()->data_type()));

    _is_quantized = is_data_type_quantized_asymmetric(input->info()->data_type());

    if(biases != nullptr)
    {
        if(_is_quantized)
        {
            ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        }
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, biases);
        ARM_COMPUTE_ERROR_ON(!weights_info.are_reshaped() && biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    const DataType dt = input->info()->data_type();

    // Set the GPU target for matrix multiply
    _mm_kernel.set_target(CLScheduler::get().target());

    _append_bias          = (biases != nullptr) && (!_is_quantized);
    _are_weights_reshaped = weights_info.are_reshaped();

    const unsigned   bias_element  = (_append_bias) ? 1 : 0;
    const ICLTensor *biases_to_use = (_append_bias) ? biases : nullptr;

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
    const bool run_interleaved      = (!_is_fully_connected_convolution && !_is_quantized);

    unsigned int mat_weights_cols = weights->info()->dimension(3);
    unsigned int mat_weights_rows = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2) + bias_element;

    // Reshape weights if needed
    if(_are_weights_reshaped)
    {
        if(_is_fully_connected_convolution || _is_quantized)
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
        if(_is_fully_connected_convolution || _is_quantized)
        {
            // Create tensor to store the reshaped weights
            TensorShape shape_wr(mat_weights_cols, mat_weights_rows);
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
        _weights_reshaped.info()->set_quantization_info(weights->info()->quantization_info());
        weights = &_weights_reshaped;
    }

    // Create tensor to store im2col reshaped inputs
    const unsigned int mat_input_cols = mat_weights_rows;
    const unsigned int mat_input_rows = conv_w * conv_h;
    TensorShape        shape_im2col   = input->info()->tensor_shape();
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);
    shape_im2col.set(2, 1);
    TensorInfo im2col_reshaped_info(shape_im2col, 1, dt, input->info()->fixed_point_position());
    im2col_reshaped_info.set_quantization_info(input->info()->quantization_info());
    _input_im2col_reshaped.allocator()->init(im2col_reshaped_info);
    _memory_group.manage(&_input_im2col_reshaped);

    // Create tensor (interleave) to prepare input tensor for GEMM
    if(run_interleaved)
    {
        TensorShape shape_interleaved = shape_im2col;
        shape_interleaved.set(0, shape_interleaved.x() * 4);
        shape_interleaved.set(1, std::ceil(shape_interleaved.y() / 4.f));
        TensorInfo interleaved_info(shape_interleaved, 1, dt, input->info()->fixed_point_position());
        interleaved_info.set_quantization_info(input->info()->quantization_info());
        _input_interleaved_reshaped.allocator()->init(interleaved_info);
        _memory_group.manage(&_input_interleaved_reshaped);
    }

    // Create GEMM output tensor
    TensorShape shape_gemm = _input_im2col_reshaped.info()->tensor_shape();
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    const DataType gemm_data_type = _is_quantized ? DataType::S32 : dt;
    // GEMM output should be S32 for acquiring raw integer accumulator without quantized postprocessing for quantized asymmetric input.
    TensorInfo info_gemm(shape_gemm, 1, gemm_data_type, input->info()->fixed_point_position());
    info_gemm.set_quantization_info(output->info()->quantization_info());
    _gemm_output.allocator()->init(info_gemm);
    _memory_group.manage(&_gemm_output);

    // Configure kernels
    _input_im2col_kernel.set_target(CLScheduler::get().target());
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

    // Configure output stage for quantized case
    if(_is_quantized)
    {
        float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output->info()->quantization_info().scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        _gemmlowp_output_stage.configure(&_gemm_output, biases, &_tmp_output, output_multiplier, output_shift, output->info()->quantization_info().offset);
        _gemm_output.allocator()->allocate();
    }

    // Configure Col2Im
    _output_col2im_kernel.set_target(CLScheduler::get().target());
    _output_col2im_kernel.configure(_is_quantized ? &_tmp_output : &_gemm_output, output, std::make_pair(conv_w, conv_h));
    if(_is_quantized)
    {
        _tmp_output.allocator()->allocate();
    }
    else
    {
        _gemm_output.allocator()->allocate();
    }

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(0) != conv_w) || (output->info()->dimension(1) != conv_h), "Output shape does not match the expected one");

    // Allocate intermediate tensor
    if(!_are_weights_reshaped)
    {
        _weights_reshaped.allocator()->allocate();
    }
}

void CLConvolutionLayer::run()
{
    // Run weights reshaping (Runs once for every configure)
    if(!_are_weights_reshaped)
    {
        _are_weights_reshaped = true;
        _reshape_weights.run();
    }

    _memory_group.acquire();

    // Run im2col
    CLScheduler::get().enqueue(_input_im2col_kernel);

    if(!_is_fully_connected_convolution && !_is_quantized)
    {
        // Run interleave4x4
        CLScheduler::get().enqueue(_input_interleave_kernel);
    }

    // Runs matrix multiply on reshaped matrices
    if(_is_quantized)
    {
        _mm_gemmlowp.run();
    }
    else
    {
        CLScheduler::get().enqueue(_mm_kernel);
    }

    // Run output stage for quantized case
    if(_is_quantized)
    {
        _gemmlowp_output_stage.run();
    }

    // Reshape output matrix
    CLScheduler::get().enqueue(_output_col2im_kernel, false);

    _memory_group.release();
}
