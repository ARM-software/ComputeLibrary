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
#include "arm_compute/graph/nodes/FullyConnectedLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Logger.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
TensorShape calculate_fullyconnected_layer_output_shape(const TensorShape &input_shape, unsigned int output_neurons)
{
    // Note: Only 1D batch space is supported at the moment
    unsigned int batches = input_shape[1];
    if(input_shape.num_dimensions() > 2)
    {
        batches = input_shape[3];
    }
    return TensorShape(output_neurons, batches);
}
template <typename FullyConnectedType, typename TensorType, TargetHint target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output)
{
    bool weights_are_loaded = weights.tensor() != nullptr;
    bool biases_are_loaded  = biases.tensor() != nullptr;

    auto conv = arm_compute::support::cpp14::make_unique<FullyConnectedType>();
    conv->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(weights.set_target(target_hint)),
        dynamic_cast<TensorType *>(biases.set_target(target_hint)),
        dynamic_cast<TensorType *>(output));
    if(!weights_are_loaded)
    {
        weights.allocate_and_fill_if_needed();
    }
    if(!biases_are_loaded)
    {
        biases.allocate_and_fill_if_needed();
    }

    return std::move(conv);
}

template <TargetHint                    target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::OPENCL>(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output)
{
    return instantiate_function<arm_compute::CLFullyConnectedLayer, arm_compute::CLTensor, TargetHint::OPENCL>(input, weights, biases, output);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::NEON>(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output)
{
    return instantiate_function<arm_compute::NEFullyConnectedLayer, arm_compute::Tensor, TargetHint::NEON>(input, weights, biases, output);
}
} // namespace

std::unique_ptr<arm_compute::IFunction> FullyConnectedLayer::instantiate_node(GraphContext &ctx, ITensor *input, ITensor *output)
{
    if(_weights.tensor() == nullptr)
    {
        unsigned int num_weights    = 1;
        unsigned int num_dimensions = input->info()->num_dimensions();
        // Ignore the batch dimension if there is one:
        if(num_dimensions == 2 || num_dimensions == 4)
        {
            num_dimensions--;
        }
        for(unsigned int i = 0; i < num_dimensions; i++)
        {
            num_weights *= input->info()->dimension(i);
        }
        _weights.set_info(TensorInfo(TensorShape(num_weights, _num_neurons), input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position()));
    }
    if(_biases.tensor() == nullptr)
    {
        _biases.set_info(TensorInfo(TensorShape(_num_neurons), input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position()));
    }

    // Auto configure output
    arm_compute::auto_init_if_empty(*output->info(),
                                    calculate_fullyconnected_layer_output_shape(input->info()->tensor_shape(), _num_neurons),
                                    input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position());

    std::unique_ptr<arm_compute::IFunction> func;
    _target_hint = ctx.hints().target_hint();

    if(_target_hint == TargetHint::OPENCL)
    {
        func = instantiate<TargetHint::OPENCL>(input, _weights, _biases, output);
        ARM_COMPUTE_LOG("Instantiating CLFullyConnectedLayer");
    }
    else
    {
        func = instantiate<TargetHint::NEON>(input, _weights, _biases, output);
        ARM_COMPUTE_LOG("Instantiating NEFullyConnectedLayer");
    }

    ARM_COMPUTE_LOG(" Type: " << input->info()->data_type()
                    << " Input Shape: " << input->info()->tensor_shape()
                    << " Weights shape: " << _weights.info().tensor_shape()
                    << " Biases Shape: " << _biases.info().tensor_shape()
                    << " Output Shape: " << output->info()->tensor_shape()
                    << std::endl);

    return func;
}
