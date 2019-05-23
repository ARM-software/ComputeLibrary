/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCFullyConnectedLayer.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "support/ToolchainSupport.h"

#include <algorithm>

using namespace arm_compute;

void GCFullyConnectedLayerReshapeWeights::configure(const IGCTensor *input, IGCTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<GCTransposeKernel>();
    k->configure(input, output);
    _kernel = std::move(k);
}

GCFullyConnectedLayer::GCFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _im2col_kernel(), _reshape_weights_kernel(), _mm_kernel(), _accumulate_biases_kernel(), _im2col_output(), _reshape_weights_output(),
      _original_weights(nullptr), _are_weights_reshaped(true), _is_fc_after_conv(true), _accumulate_biases(false)
{
}

void GCFullyConnectedLayer::configure_conv_fc(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON((weights->info()->dimension(1) != (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))));

    const DataType dt = input->info()->data_type();

    // If the fully connected layer is called after a convolution layer, the input tensor must be linearized

    // Initialize output tensor for im2col
    TensorShape shape_im2col;
    shape_im2col.set(0, input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2));
    shape_im2col.set(1, input->info()->dimension(3));
    shape_im2col.set(2, input->info()->dimension(4));
    shape_im2col.set(3, input->info()->dimension(5));
    _im2col_output.allocator()->init(TensorInfo(shape_im2col, 1, dt));

    // Configure im2col kernel
    _memory_group.manage(&_im2col_output);
    _im2col_kernel.configure(input, &_im2col_output, Size2D(1, 1), PadStrideInfo(1, 1, 0, 0), false);

    // Configure matrix multiply kernel
    _mm_kernel.configure(&_im2col_output, weights, output, 1.0f, false);

    // Allocate the output tensor for im2col once all the configure methods have been called
    _im2col_output.allocator()->allocate();
}

void GCFullyConnectedLayer::configure_fc_fc(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != weights->info()->dimension(1));

    // Configure matrix multiply kernel
    _mm_kernel.configure(input, weights, output, 1.0f, false);
}

void GCFullyConnectedLayer::configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output,
                                      FullyConnectedLayerInfo fc_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 2);

    _original_weights     = weights;
    _are_weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    _is_fc_after_conv     = true;
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

    const IGCTensor *weights_to_use = weights;

    if(!_are_weights_reshaped)
    {
        weights_to_use = &_reshape_weights_output;

        // Reshape the weights
        _reshape_weights_kernel.configure(weights, &_reshape_weights_output);
    }

    // Check if we have a fully connected layer with batches
    const bool is_batched_fc_layer = output->info()->dimension(1) > 1;

    if(is_batched_fc_layer)
    {
        _is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                                                                  input->info()->tensor_shape().cend(),
                                                                                  output->info()->tensor_shape().cbegin() + 1));
    }
    else
    {
        _is_fc_after_conv = input->info()->num_dimensions() > 1;
    }

    if(_is_fc_after_conv)
    {
        // Fully Connected layer after a Convolution Layer without batches
        configure_conv_fc(input, weights_to_use, output);
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        configure_fc_fc(input, weights_to_use, output);
    }

    ARM_COMPUTE_ERROR_ON(fc_info.retain_internal_weights && _reshape_weights_output.gc_buffer() == 0);
    _are_weights_reshaped = _are_weights_reshaped || fc_info.retain_internal_weights;
}

void GCFullyConnectedLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Linearize input if it comes from a convolutional layer
    if(_is_fc_after_conv)
    {
        GCScheduler::get().dispatch(_im2col_kernel, false);
    }

    if(!_are_weights_reshaped || _is_fc_after_conv)
    {
        GCScheduler::get().memory_barrier();
    }

    // Run matrix multiply
    GCScheduler::get().dispatch(_mm_kernel, !_accumulate_biases);

    // Accumulate biases if provided
    if(_accumulate_biases)
    {
        GCScheduler::get().memory_barrier();

        GCScheduler::get().dispatch(_accumulate_biases_kernel);
    }
}

void GCFullyConnectedLayer::prepare()
{
    // Reshape of the weights (happens only once)
    if(!_are_weights_reshaped)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        // Run reshape weights kernel and mark weights as unused
        _reshape_weights_output.allocator()->allocate();
        _reshape_weights_kernel.run();

        // Mark original weights tensor as unused
        _original_weights->mark_as_unused();

        _are_weights_reshaped = true;
    }
}