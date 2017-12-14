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
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <algorithm>
#include <cmath>

namespace arm_compute
{
NEFullyConnectedLayerReshapeWeights::NEFullyConnectedLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _transpose_kernel(), _transpose1xW_kernel(), _transpose_output(), _transpose_weights(false), _is_batched_fc_layer(false)
{
}

void NEFullyConnectedLayerReshapeWeights::configure(const ITensor *input, ITensor *output, bool transpose_weights, bool is_batched_fc_layer)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() > 2);
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    ARM_COMPUTE_ERROR_ON(!transpose_weights && !is_batched_fc_layer);

    const DataType data_type            = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();

    _transpose_weights   = transpose_weights;
    _is_batched_fc_layer = is_batched_fc_layer;

    // Check if we need to transpose the weights
    if(_transpose_weights)
    {
        if(_is_batched_fc_layer)
        {
            // Initialize the output tensor for transpose
            TensorShape shape_transposed(input->info()->dimension(1), input->info()->dimension(0));
            _transpose_output.allocator()->init(TensorInfo(shape_transposed, 1, data_type, fixed_point_position));
            _memory_group.manage(&_transpose_output);
            _transpose_kernel.configure(input, &_transpose_output);

            // Configure transpose 1xW kernel
            _transpose1xW_kernel.configure(&_transpose_output, output);

            // Allocate temporary tensor used for transposing the weights
            _transpose_output.allocator()->allocate();
        }
        else
        {
            _transpose_kernel.configure(input, output);
        }
    }
    else
    {
        if(_is_batched_fc_layer)
        {
            // Configure transpose 1xW kernel
            _transpose1xW_kernel.configure(input, output);
        }
        else
        {
            ARM_COMPUTE_ERROR("Configuration transpose_weights=false & is_batched_fc_layer=false not supported");
        }
    }
}

void NEFullyConnectedLayerReshapeWeights::run()
{
    _memory_group.acquire();

    if(_transpose_weights)
    {
        NEScheduler::get().schedule(&_transpose_kernel, Window::DimY);
    }

    if(_is_batched_fc_layer)
    {
        NEScheduler::get().schedule(&_transpose1xW_kernel, Window::DimY);
    }

    _memory_group.release();
}

NEFullyConnectedLayer::NEFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _im2col_kernel(), _reshape_weights_kernel(), _interleave4x4_kernel(), _mm_kernel(), _accumulate_biases_kernel(), _im2col_output(), _interleave4x4_output(),
      _reshape_weights_output(), _are_weights_reshaped(false), _is_batched_fc_layer(false), _linearize_input(false), _accumulate_biases(false)
{
}

void NEFullyConnectedLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, bool transpose_weights, bool are_weights_reshaped)
{
    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    // Expected shape before transpose and reshaping
    // Input: In x B (In and B can be multi-dimensional)
    // Weights: flat(In) x Out
    // Biases: Out
    // Output: Out x B (B can be multi-dimensional)

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, weights, output);

    const DataType data_type            = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();
    const int      num_batch_dimensions = std::max(0, static_cast<int>(output->info()->tensor_shape().num_dimensions()) - 1);
    const int      num_input_dimensions = input->info()->tensor_shape().num_dimensions() - num_batch_dimensions;
    const size_t   linear_input_size    = input->info()->tensor_shape().total_size_lower(num_input_dimensions);

    _linearize_input      = (input->info()->tensor_shape().x() != linear_input_size) || (num_input_dimensions > 1 && linear_input_size == 1);
    _are_weights_reshaped = are_weights_reshaped;
    _accumulate_biases    = biases != nullptr;
    _is_batched_fc_layer  = num_batch_dimensions > 0;

    // Check if number of batches match
    ARM_COMPUTE_ERROR_ON(input->info()->tensor_shape().total_size_upper(num_input_dimensions) != output->info()->tensor_shape().total_size_upper(1));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 2);

    const size_t   interleave_width = 16 / input->info()->element_size();
    const ITensor *weights_to_use   = weights;

    if(!are_weights_reshaped && (transpose_weights || _is_batched_fc_layer))
    {
        weights_to_use = &_reshape_weights_output;

        TensorShape reshaped_weights_shape(weights->info()->tensor_shape());

        // Transpose weights if the user hasn't done it
        if(transpose_weights)
        {
            const size_t shape_x = reshaped_weights_shape.x();
            reshaped_weights_shape.set(0, reshaped_weights_shape.y());
            reshaped_weights_shape.set(1, shape_x);
        }

        // If the we run multiple batches we need 1xW transpose, too.
        if(_is_batched_fc_layer)
        {
            const float shape_x = reshaped_weights_shape.x();
            reshaped_weights_shape.set(0, reshaped_weights_shape.y() * interleave_width);
            reshaped_weights_shape.set(1, static_cast<unsigned int>(std::ceil(shape_x / interleave_width)));
        }

        _reshape_weights_output.allocator()->init(TensorInfo(reshaped_weights_shape, 1, data_type, fixed_point_position));

        // Reshape the weights
        _reshape_weights_kernel.configure(weights, &_reshape_weights_output, transpose_weights, _is_batched_fc_layer);
    }

    // Check correct shape of weights
    if(_is_batched_fc_layer)
    {
        // Transpose + Transpose1xW
        ARM_COMPUTE_ERROR_ON(weights_to_use->info()->tensor_shape().x() != linear_input_size * interleave_width);
        ARM_COMPUTE_ERROR_ON(weights_to_use->info()->tensor_shape().y() != static_cast<unsigned int>(std::ceil(static_cast<float>(output->info()->tensor_shape().x()) / interleave_width)));
    }
    else
    {
        // Transpose
        ARM_COMPUTE_ERROR_ON(weights_to_use->info()->tensor_shape().x() != output->info()->tensor_shape().x());
        ARM_COMPUTE_ERROR_ON(weights_to_use->info()->tensor_shape().y() != linear_input_size);
    }

    const ITensor *multiply_input = input;

    if(_linearize_input)
    {
        TensorShape shape_im2col(input->info()->tensor_shape());
        shape_im2col.collapse(num_input_dimensions);
        _im2col_output.allocator()->init(TensorInfo(shape_im2col, 1, data_type, fixed_point_position));

        // Configure im2col kernel
        _memory_group.manage(&_im2col_output);
        _im2col_kernel.configure(input, &_im2col_output, Size2D(1, 1), PadStrideInfo(1, 1, 0, 0), false);

        multiply_input = &_im2col_output;
    }

    if(_is_batched_fc_layer)
    {
        TensorShape shape_interleaved(multiply_input->info()->tensor_shape());
        shape_interleaved.set(0, shape_interleaved.x() * 4);
        shape_interleaved.set(1, std::ceil(shape_interleaved.y() / 4.f));
        _interleave4x4_output.allocator()->init(TensorInfo(shape_interleaved, 1, data_type, fixed_point_position));

        // Configure interleave4x4 kernel
        _memory_group.manage(&_interleave4x4_output);
        _interleave4x4_kernel.configure(multiply_input, &_interleave4x4_output);

        multiply_input = &_interleave4x4_output;
    }

    // Configure matrix multiply kernel
    _mm_kernel.configure(multiply_input, weights_to_use, output, 1.0f);

    if(_accumulate_biases)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->tensor_shape().x() != output->info()->tensor_shape().x());

        // Configure accumulate biases kernel
        _accumulate_biases_kernel.configure(output, biases);
    }

    // Allocate the transpose tensor if the are_weights_reshaped flag is false and once all the configure methods have been called
    if(!are_weights_reshaped && (transpose_weights || _is_batched_fc_layer))
    {
        // Allocate the tensor for the weights reshaped
        _reshape_weights_output.allocator()->allocate();
    }

    if(_linearize_input)
    {
        _im2col_output.allocator()->allocate();
    }

    if(_is_batched_fc_layer)
    {
        _interleave4x4_output.allocator()->allocate();
    }
}

void NEFullyConnectedLayer::run()
{
    // Reshape of the weights (happens only once)
    if(!_are_weights_reshaped)
    {
        _are_weights_reshaped = true;
        _reshape_weights_kernel.run();
    }

    _memory_group.acquire();

    // Linearize input if it comes from a convolutional layer
    if(_linearize_input)
    {
        NEScheduler::get().schedule(&_im2col_kernel, Window::DimY);
    }

    // Interleave input
    if(_is_batched_fc_layer)
    {
        NEScheduler::get().schedule(&_interleave4x4_kernel, Window::DimY);
    }

    // Run matrix multiply
    NEScheduler::get().schedule(&_mm_kernel, _is_batched_fc_layer ? Window::DimY : Window::DimX);

    // Accumulate biases if provided
    if(_accumulate_biases)
    {
        NEScheduler::get().schedule(&_accumulate_biases_kernel, Window::DimY);
    }

    _memory_group.release();
}
} // namespace arm_compute
