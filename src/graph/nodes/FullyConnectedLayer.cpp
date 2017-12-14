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

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"
#include "support/ToolchainSupport.h"

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
} // namespace

std::unique_ptr<arm_compute::IFunction> FullyConnectedLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);

    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();
    _target_hint              = ctx.hints().target_hint();

    if(_weights.tensor() == nullptr)
    {
        unsigned int num_weights    = 1;
        unsigned int num_dimensions = in->info()->num_dimensions();
        // Ignore the batch dimension if there is one:
        if(num_dimensions == 2 || num_dimensions == 4)
        {
            num_dimensions--;
        }
        for(unsigned int i = 0; i < num_dimensions; i++)
        {
            num_weights *= in->info()->dimension(i);
        }
        _weights.set_info(TensorInfo(TensorShape(num_weights, _num_neurons), in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position()));
    }
    if(_biases.tensor() == nullptr)
    {
        _biases.set_info(TensorInfo(TensorShape(_num_neurons), in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position()));
    }

    // Auto configure output
    arm_compute::auto_init_if_empty(*out->info(),
                                    calculate_fullyconnected_layer_output_shape(in->info()->tensor_shape(), _num_neurons),
                                    in->info()->num_channels(), in->info()->data_type(), in->info()->fixed_point_position());

    bool weights_are_loaded = _weights.tensor() != nullptr;
    bool biases_are_loaded  = _biases.tensor() != nullptr;

    // Create node context
    NodeContext node_ctx(OperationType::FullyConnectedLayer);
    node_ctx.set_target(_target_hint);
    node_ctx.add_input(in);
    node_ctx.add_input(_weights.set_target(_target_hint));
    node_ctx.add_input(_biases.set_target(_target_hint));
    node_ctx.add_output(out);

    // Configure operation
    auto func = OperationRegistry::get().find_operation(OperationType::FullyConnectedLayer, _target_hint)->configure(node_ctx);

    // Fill biases
    if(!weights_are_loaded)
    {
        _weights.allocate_and_fill_if_needed();
    }
    if(!biases_are_loaded)
    {
        _biases.allocate_and_fill_if_needed();
    }

    // Get function
    return func;
}
