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
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <algorithm>
#include <cmath>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

NEFullyConnectedLayerReshapeWeights::NEFullyConnectedLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _transpose_kernel(), _transpose1xW_kernel(), _transpose_output(), _transpose_weights(false), _is_batched_fc_layer(false)
{
}

void NEFullyConnectedLayerReshapeWeights::configure(const ITensor *input, ITensor *output, bool transpose_weights, bool is_batched_fc_layer)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(NEFullyConnectedLayerReshapeWeights::validate(input->info(), output->info(), transpose_weights, is_batched_fc_layer));

    _transpose_weights   = transpose_weights;
    _is_batched_fc_layer = is_batched_fc_layer;

    // Check if we need to transpose the weights
    if(_transpose_weights)
    {
        if(_is_batched_fc_layer)
        {
            // Initialize the output tensor for transpose
            _transpose_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_transposed_shape(*input->info())));
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
    }
}

Status NEFullyConnectedLayerReshapeWeights::validate(const ITensorInfo *input, const ITensorInfo *output, bool transpose_weights, bool is_batched_fc_layer)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!transpose_weights && !is_batched_fc_layer, "Configuration transpose_weights=false & is_batched_fc_layer=false not supported");

    if(transpose_weights)
    {
        if(is_batched_fc_layer)
        {
            std::unique_ptr<ITensorInfo> use_output = output->clone();
            use_output->set_is_resizable(true).reset_padding().set_tensor_shape(compute_transposed_shape(*input));

            ARM_COMPUTE_RETURN_ON_ERROR(NETransposeKernel::validate(input, use_output.get()));
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(use_output.get(), output));
        }
        else
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NETransposeKernel::validate(input, output));
        }
    }
    else
    {
        if(is_batched_fc_layer)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(input, output));
        }
    }

    return Status{};
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
      _reshape_weights_output(), _are_weights_reshaped(false), _is_batched_fc_layer(false), _linearize_input(false), _accumulate_biases(false), _original_weights(nullptr)
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
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(NEFullyConnectedLayer::validate(input->info(),
                                                               weights->info(),
                                                               biases != nullptr ? biases->info() : nullptr,
                                                               output->info(),
                                                               transpose_weights,
                                                               are_weights_reshaped));

    const int    num_batch_dimensions = std::max(0, static_cast<int>(output->info()->tensor_shape().num_dimensions()) - 1);
    const int    num_input_dimensions = input->info()->tensor_shape().num_dimensions() - num_batch_dimensions;
    const size_t linear_input_size    = input->info()->tensor_shape().total_size_lower(num_input_dimensions);

    _original_weights     = weights;
    _linearize_input      = (input->info()->tensor_shape().x() != linear_input_size) || (num_input_dimensions > 1 && linear_input_size == 1);
    _are_weights_reshaped = are_weights_reshaped;
    _accumulate_biases    = biases != nullptr;
    _is_batched_fc_layer  = num_batch_dimensions > 0;

    const size_t   interleave_width = 16 / input->info()->element_size();
    const ITensor *weights_to_use   = weights;

    if(!are_weights_reshaped && (transpose_weights || _is_batched_fc_layer))
    {
        weights_to_use = &_reshape_weights_output;

        _reshape_weights_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_fully_connected_reshaped_weights_shape(weights->info(),
                                                  transpose_weights,
                                                  _is_batched_fc_layer, interleave_width)));

        // Reshape the weights
        _reshape_weights_kernel.configure(weights, &_reshape_weights_output, transpose_weights, _is_batched_fc_layer);
    }

    const ITensor *multiply_input = input;

    if(_linearize_input)
    {
        _im2col_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_im2col_fc_shape(input->info(), num_input_dimensions)));

        // Configure im2col kernel
        _memory_group.manage(&_im2col_output);
        _im2col_kernel.configure(input, &_im2col_output, Size2D(1, 1), PadStrideInfo(1, 1, 0, 0), false, true);

        multiply_input = &_im2col_output;
    }

    int m = multiply_input->info()->dimension(1);
    int k = multiply_input->info()->dimension(0);

    if(_is_batched_fc_layer)
    {
        _interleave4x4_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_interleaved_shape(*multiply_input->info())));

        // Configure interleave4x4 kernel
        _memory_group.manage(&_interleave4x4_output);
        _interleave4x4_kernel.configure(multiply_input, &_interleave4x4_output);

        multiply_input = &_interleave4x4_output;
    }

    // Configure matrix multiply kernel
    _mm_kernel.configure(multiply_input, weights_to_use, output, 1.0f, _is_batched_fc_layer, GEMMReshapeInfo(m, 0 /* no transpose */, k));

    if(_accumulate_biases)
    {
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

Status NEFullyConnectedLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, bool transpose_weights, bool are_weights_reshaped)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, weights, output);

    const int    num_batch_dimensions = std::max(0, static_cast<int>(output->tensor_shape().num_dimensions()) - 1);
    const int    num_input_dimensions = input->tensor_shape().num_dimensions() - num_batch_dimensions;
    const size_t linear_input_size    = input->tensor_shape().total_size_lower(num_input_dimensions);

    const bool linearize_input     = (input->tensor_shape().x() != linear_input_size) || (num_input_dimensions > 1 && linear_input_size == 1);
    const bool accumulate_biases   = biases != nullptr;
    const bool is_batched_fc_layer = num_batch_dimensions > 0;

    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().total_size_upper(num_input_dimensions) != output->tensor_shape().total_size_upper(1));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);

    const size_t                 interleave_width       = 16 / input->element_size();
    const ITensorInfo           *weights_to_use         = weights;
    std::unique_ptr<ITensorInfo> reshape_weights_output = input->clone();

    if(!are_weights_reshaped && (transpose_weights || is_batched_fc_layer))
    {
        reshape_weights_output->set_tensor_shape(compute_fully_connected_reshaped_weights_shape(weights, transpose_weights, is_batched_fc_layer, interleave_width));

        ARM_COMPUTE_RETURN_ON_ERROR(NEFullyConnectedLayerReshapeWeights::validate(weights, reshape_weights_output.get(), transpose_weights, is_batched_fc_layer));

        weights_to_use = reshape_weights_output.get();
    }

    // Check correct shape of weights
    if(is_batched_fc_layer)
    {
        // Transpose + Transpose1xW
        ARM_COMPUTE_RETURN_ERROR_ON(weights_to_use->tensor_shape().x() != linear_input_size * interleave_width);
        ARM_COMPUTE_RETURN_ERROR_ON(weights_to_use->tensor_shape().y() != static_cast<unsigned int>(std::ceil(static_cast<float>(output->tensor_shape().x()) / interleave_width)));
    }
    else
    {
        // Transpose
        ARM_COMPUTE_RETURN_ERROR_ON(weights_to_use->tensor_shape().x() != output->tensor_shape().x());
        ARM_COMPUTE_RETURN_ERROR_ON(weights_to_use->tensor_shape().y() != linear_input_size);
    }

    const ITensorInfo           *multiply_input       = input;
    std::unique_ptr<ITensorInfo> im2col_output        = input->clone();
    std::unique_ptr<ITensorInfo> interleave4x4_output = input->clone();

    if(linearize_input)
    {
        im2col_output->set_tensor_shape(compute_im2col_fc_shape(input, num_input_dimensions));

        ARM_COMPUTE_RETURN_ON_ERROR(NEIm2ColKernel::validate(input, im2col_output.get(), Size2D(1, 1), PadStrideInfo(1, 1, 0, 0), false, true));

        multiply_input = im2col_output.get();
    }

    int m = multiply_input->dimension(1);
    int k = multiply_input->dimension(0);

    if(is_batched_fc_layer)
    {
        interleave4x4_output->set_tensor_shape(compute_interleaved_shape(*multiply_input));

        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMInterleave4x4Kernel::validate(multiply_input, interleave4x4_output.get()));

        multiply_input = interleave4x4_output.get();
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixMultiplyKernel::validate(multiply_input, weights_to_use, output, 1.0f, is_batched_fc_layer, GEMMReshapeInfo(m, 0 /* no transpose */, k)));

    if(accumulate_biases)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->tensor_shape().x() != output->tensor_shape().x());

        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixAccumulateBiasesKernel::validate(output, biases));
    }

    return Status{};
}

void NEFullyConnectedLayer::run()
{
    // Reshape of the weights (happens only once)
    if(!_are_weights_reshaped)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        _are_weights_reshaped = true;
        _reshape_weights_kernel.run();

        // Mark original weights tensor as unused
        _original_weights->mark_as_unused();
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
