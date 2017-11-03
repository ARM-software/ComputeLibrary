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
#include "arm_compute/graph/nodes/FlattenLayer.h"

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::graph;

std::unique_ptr<arm_compute::IFunction> FlattenLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);

    _target_hint              = ctx.hints().target_hint();
    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();

    // Auto configure output
    TensorShape tensor_shape = in->info()->tensor_shape();
    tensor_shape.collapse(in->info()->num_dimensions());
    arm_compute::auto_init_if_empty(*out->info(), tensor_shape, 1, in->info()->data_type(), in->info()->fixed_point_position());

    // Create node context
    NodeContext node_ctx(OperationType::FlattenLayer);
    node_ctx.set_target(_target_hint);
    node_ctx.add_input(in);
    node_ctx.add_output(out);

    // Get function
    return OperationRegistry::get().find_operation(OperationType::FlattenLayer, _target_hint)->configure(node_ctx);
}