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
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <algorithm>
#include <cmath>

using namespace arm_compute;

CLFullyConnectedLayerReshapeWeights::CLFullyConnectedLayerReshapeWeights()
    : _transpose_kernel(), _transpose1xW_kernel(), _transpose_output(), _transpose_weights(false), _is_batched_fc_layer(false)
{
}

void CLFullyConnectedLayerReshapeWeights::configure(const ICLTensor *input, ICLTensor *output, bool transpose_weights, bool is_batched_fc_layer)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() != 2);
    ARM_COMPUTE_ERROR_ON((transpose_weights == false) && (is_batched_fc_layer == false));

    const DataType dt                   = input->info()->data_type();
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
            _transpose_output.allocator()->init(TensorInfo(shape_transposed, 1, dt, fixed_point_position));
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

void CLFullyConnectedLayerReshapeWeights::run()
{
    if(_transpose_weights)
    {
        CLScheduler::get().enqueue(_transpose_kernel, _is_batched_fc_layer);
    }
    if(_is_batched_fc_layer)
    {
        CLScheduler::get().enqueue(_transpose1xW_kernel);
    }
}

CLFullyConnectedLayer::CLFullyConnectedLayer()
    : _im2col_kernel(), _reshape_weights_kernel(), _interleave4x4_kernel(), _mm_kernel(), _accumulate_biases_kernel(), _im2col_output(), _interleave4x4_output(), _reshape_weights_output(),
      _are_weights_reshaped(true), _is_fc_after_conv(true), _is_batched_fc_layer(false), _accumulate_biases(false)
{
}

void CLFullyConnectedLayer::configure_conv_fc_wb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2) * (16 / weights->info()->element_size())));

    const DataType dt                   = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();

    // If the fully connected layer is called after a convolution layer, the input tensor must be linearized

    // Initialize output tensor for im2col
    TensorShape shape_im2col;
    shape_im2col.set(0, input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2));
    shape_im2col.set(1, input->info()->dimension(3));
    shape_im2col.set(2, input->info()->dimension(4));
    shape_im2col.set(3, input->info()->dimension(5));
    _im2col_output.allocator()->init(TensorInfo(shape_im2col, 1, dt, fixed_point_position));

    // Initialize output tensor for interleave 4x4
    TensorShape shape_interleaved = _im2col_output.info()->tensor_shape();
    shape_interleaved.set(0, shape_interleaved.x() * 4);
    shape_interleaved.set(1, std::ceil(static_cast<float>(shape_interleaved.y()) / 4));
    _interleave4x4_output.allocator()->init(TensorInfo(shape_interleaved, 1, dt, fixed_point_position));

    // Configure im2col kernel
    _im2col_kernel.configure(input, &_im2col_output, Size2D(1, 1), PadStrideInfo(1, 1, 0, 0), false);

    // Configure interleave4x4 kernel
    _interleave4x4_kernel.configure(&_im2col_output, &_interleave4x4_output);

    // Configure matrix multiply kernel
    _mm_kernel.configure(&_interleave4x4_output, weights, output, 1.0f);

    // Allocate the tensors once all the configure methods have been called
    _im2col_output.allocator()->allocate();
    _interleave4x4_output.allocator()->allocate();
}

void CLFullyConnectedLayer::configure_fc_fc_wb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output)
{
    const DataType dt                   = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();

    // Initialize output tensor for interleave 4x4
    TensorShape shape_interleaved = input->info()->tensor_shape();
    shape_interleaved.set(0, shape_interleaved.x() * 4);
    shape_interleaved.set(1, std::ceil(static_cast<float>(shape_interleaved.y()) / 4));
    _interleave4x4_output.allocator()->init(TensorInfo(shape_interleaved, 1, dt, fixed_point_position));

    // Configure interleave4x4 kernel
    _interleave4x4_kernel.configure(input, &_interleave4x4_output);

    // Configure matrix multiply kernel
    _mm_kernel.configure(&_interleave4x4_output, weights, output, 1.0f);

    // Allocate the tensors once all the configure methods have been called
    _interleave4x4_output.allocator()->allocate();
}

void CLFullyConnectedLayer::configure_conv_fc_nb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON((weights->info()->dimension(1) != (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))));

    const DataType dt                   = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();

    // If the fully connected layer is called after a convolution layer, the input tensor must be linearized

    // Initialize output tensor for im2col
    TensorShape shape_im2col;
    shape_im2col.set(0, input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2));
    shape_im2col.set(1, 1);
    _im2col_output.allocator()->init(TensorInfo(shape_im2col, 1, dt, fixed_point_position));

    // Configure im2col kernel
    _im2col_kernel.configure(input, &_im2col_output, Size2D(1, 1), PadStrideInfo(1, 1, 0, 0), false);

    // Configure matrix multiply kernel
    _mm_kernel.configure(&_im2col_output, weights, output, 1.0f);

    // Allocate the output tensor for im2col once all the configure methods have been called
    _im2col_output.allocator()->allocate();
}

void CLFullyConnectedLayer::configure_fc_fc_nb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != weights->info()->dimension(1));

    // Configure matrix multiply kernel
    _mm_kernel.configure(input, weights, output, 1.0f);
}

void CLFullyConnectedLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, bool transpose_weights, bool are_weights_reshaped)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() != 2);

    const DataType dt                   = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();

    _are_weights_reshaped = are_weights_reshaped;
    _is_fc_after_conv     = true;
    _is_batched_fc_layer  = false;
    _accumulate_biases    = false;

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);

        _accumulate_biases = true;

        // Configure accumulate biases kernel
        _accumulate_biases_kernel.configure(output, biases);
    }

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    // Check if we have a fully connected layer with batches
    _is_batched_fc_layer = (output->info()->dimension(1) > 1);

    const ICLTensor *weights_to_use = weights;

    if(!are_weights_reshaped)
    {
        if((transpose_weights || _is_batched_fc_layer))
        {
            weights_to_use = &_reshape_weights_output;

            if(transpose_weights)
            {
                if(_is_batched_fc_layer)
                {
                    const float transpose_width = 16.0f / input->info()->element_size();
                    TensorShape shape_wt(weights->info()->dimension(0) * static_cast<unsigned int>(transpose_width), static_cast<unsigned int>(std::ceil(weights->info()->dimension(1) / transpose_width)));
                    TensorInfo  info_wt(shape_wt, 1, dt, fixed_point_position);
                    _reshape_weights_output.allocator()->init(info_wt);
                }
                else
                {
                    TensorShape shape_wt(weights->info()->dimension(1), weights->info()->dimension(0));
                    TensorInfo  info_wt(shape_wt, 1, dt, fixed_point_position);
                    _reshape_weights_output.allocator()->init(info_wt);
                }
            }
            else
            {
                ARM_COMPUTE_ERROR_ON(!_is_batched_fc_layer);

                const float transpose_width = 16.0f / input->info()->element_size();
                TensorShape shape_wt(weights->info()->dimension(1) * static_cast<unsigned int>(transpose_width), static_cast<unsigned int>(std::ceil(weights->info()->dimension(0) / transpose_width)));
                TensorInfo  info_wt(shape_wt, 1, dt, fixed_point_position);
                _reshape_weights_output.allocator()->init(info_wt);
            }

            // Reshape the weights
            _reshape_weights_kernel.configure(weights, &_reshape_weights_output, transpose_weights, _is_batched_fc_layer);
        }
    }

    if(_is_batched_fc_layer)
    {
        _is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                                                                  input->info()->tensor_shape().cend(),
                                                                                  output->info()->tensor_shape().cbegin() + 1));

        if(_is_fc_after_conv)
        {
            // Fully Connected layer after a Convolution Layer with batches
            configure_conv_fc_wb(input, weights_to_use, output);
        }
        else
        {
            // Fully Connected layer after a Fully Connected Layer with batches
            configure_fc_fc_wb(input, weights_to_use, output);
        }
    }
    else
    {
        // In case of not batched fully connected layer, the weights will not be reshaped using transposed1xW
        _is_fc_after_conv = ((weights_to_use->info()->dimension(1)) == (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2)));

        if(_is_fc_after_conv)
        {
            // Fully Connected layer after a Convolution Layer without batches
            configure_conv_fc_nb(input, weights_to_use, output);
        }
        else
        {
            // Fully Connected layer after a Fully Connected Layer without batches
            configure_fc_fc_nb(input, weights_to_use, output);
        }
    }

    // Allocate the transpose tensor if the are_weights_reshaped flag is false and once all the configure methods have been called
    if(!are_weights_reshaped)
    {
        if(transpose_weights || _is_batched_fc_layer)
        {
            // Allocate the tensor for the weights reshaped
            _reshape_weights_output.allocator()->allocate();
        }
    }
}

void CLFullyConnectedLayer::run()
{
    // Reshape of the weights (happens only once)
    if(!_are_weights_reshaped)
    {
        _are_weights_reshaped = true;
        _reshape_weights_kernel.run();
    }

    // Linearize input if it comes from a convolutional layer
    if(_is_fc_after_conv)
    {
        CLScheduler::get().enqueue(_im2col_kernel, false);
    }

    // Interleave input
    if(_is_batched_fc_layer)
    {
        CLScheduler::get().enqueue(_interleave4x4_kernel, false);
    }

    // Run matrix multiply
    CLScheduler::get().enqueue(_mm_kernel, !_accumulate_biases);

    // Accumulate biases if provided
    if(_accumulate_biases)
    {
        CLScheduler::get().enqueue(_accumulate_biases_kernel);
    }
}
