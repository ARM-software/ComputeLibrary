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
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

#include <algorithm>

using namespace arm_compute;

void CLFullyConnectedLayerReshapeWeights::configure(const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLTransposeKernel>();
    k->configure(input, output);
    _kernel = std::move(k);
}

CLFullyConnectedLayer::CLFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _im2col_kernel(), _reshape_weights_kernel(), _mm_kernel(), _mm_gemmlowp(memory_manager), _gemmlowp_output_stage(), _accumulate_biases_kernel(), _im2col_output(),
      _gemmlowp_output(), _reshape_weights_output(), _are_weights_reshaped(true), _is_fc_after_conv(true), _accumulate_biases(false), _is_quantized(false)
{
}

void CLFullyConnectedLayer::configure_mm(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output, bool is_interleaved_transposed)
{
    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.scale, -input_quantization_info.offset));
        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.scale, -weights_quantization_info.offset));

        // Configure gemmlowp function
        _mm_gemmlowp.configure(input, weights, output);

        // Revert back QuantizatioInfo as input and weights could be used in other fully connected layers
        input->info()->set_quantization_info(input_quantization_info);
        weights->info()->set_quantization_info(weights_quantization_info);
    }
    else
    {
        // Configure matrix multiply kernel
        _mm_kernel.set_target(CLScheduler::get().target());
        _mm_kernel.configure(input, weights, output, 1.f, is_interleaved_transposed);
    }
}

void CLFullyConnectedLayer::configure_conv_fc(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON((weights->info()->dimension(1) != (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))));

    // If the fully connected layer is called after a convolution layer, the input tensor must be linearized

    // Initialize output tensor for im2col
    TensorShape shape_im2col = input->info()->tensor_shape();
    shape_im2col.collapse(3);
    _im2col_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_im2col));

    // Configure im2col kernel
    _memory_group.manage(&_im2col_output);
    _im2col_kernel.configure(input, &_im2col_output, Size2D(1, 1), PadStrideInfo(1, 1, 0, 0), false);

    // Configure matrix multiply kernel
    configure_mm(&_im2col_output, weights, output, false);

    // Allocate the output tensor for im2col once all the configure methods have been called
    _im2col_output.allocator()->allocate();
}

void CLFullyConnectedLayer::configure_fc_fc(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != weights->info()->dimension(1));

    // Configure matrix multiply kernel
    configure_mm(input, weights, output, false);
}

void CLFullyConnectedLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, bool transpose_weights, bool are_weights_reshaped)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 2);

    _are_weights_reshaped = transpose_weights ? are_weights_reshaped : true;
    _is_fc_after_conv     = true;
    _accumulate_biases    = false;
    _is_quantized         = is_data_type_quantized_asymmetric(input->info()->data_type());

    // Configure gemmlowp output
    if(_is_quantized)
    {
        _gemmlowp_output.allocator()->init(output->info()->clone()->set_is_resizable(true).reset_padding().set_data_type(DataType::S32));
    }

    // Configure accumulate biases kernel for non quantized asymmetric types
    if(biases != nullptr && !_is_quantized)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);

        _accumulate_biases = true;

        // Configure accumulate biases kernel
        _accumulate_biases_kernel.set_target(CLScheduler::get().target());
        _accumulate_biases_kernel.configure(output, biases);
    }

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    const ICLTensor *weights_to_use = weights;

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

    ICLTensor *tmp_output = (_is_quantized) ? &_gemmlowp_output : output;
    if(_is_fc_after_conv)
    {
        // Fully Connected layer after a Convolution Layer without batches
        configure_conv_fc(input, weights_to_use, tmp_output);
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        configure_fc_fc(input, weights_to_use, tmp_output);
    }

    // Configure output stage for asymmetric quantized types
    if(_is_quantized)
    {
        float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output->info()->quantization_info().scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        _gemmlowp_output_stage.configure(&_gemmlowp_output, biases, output, output_multiplier, output_shift, output->info()->quantization_info().offset);
        _gemmlowp_output.allocator()->allocate();
    }

    // Allocate the transpose tensor if the are_weights_reshaped flag is false and once all the configure methods have been called
    if(!_are_weights_reshaped)
    {
        // Allocate the tensor for the weights reshaped
        _reshape_weights_output.allocator()->allocate();
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

    _memory_group.acquire();

    // Linearize input if it comes from a convolutional layer
    if(_is_fc_after_conv)
    {
        CLScheduler::get().enqueue(_im2col_kernel, false);
    }

    // Run matrix multiply
    if(_is_quantized)
    {
        _mm_gemmlowp.run();
    }
    else
    {
        CLScheduler::get().enqueue(_mm_kernel, !_accumulate_biases);
    }

    // Accumulate biases if provided
    if(_is_quantized)
    {
        _gemmlowp_output_stage.run();
    }
    else
    {
        if(_accumulate_biases)
        {
            CLScheduler::get().enqueue(_accumulate_biases_kernel);
        }
    }

    _memory_group.release();
}
