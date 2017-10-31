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
#include "arm_compute/graph/nodes/L2NormalizeLayer.h"

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::graph;

L2NormalizeLayer::L2NormalizeLayer(unsigned int axis, float epsilon)
    : _axis(axis), _epsilon(epsilon)
{
}

std::unique_ptr<arm_compute::IFunction> L2NormalizeLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);

    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();
    _target_hint              = ctx.hints().target_hint();

    // Create node context
    NodeContext node_ctx(OperationType::L2NormalizeLayer);
    node_ctx.set_target(_target_hint);
    node_ctx.add_input(in);
    node_ctx.add_output(out);
    node_ctx.add_parameter<unsigned int>("axis", _axis);
    node_ctx.add_parameter<float>("epsilon", _epsilon);

    // Get function
    return OperationRegistry::get().find_operation(OperationType::L2NormalizeLayer, _target_hint)->configure(node_ctx);
}
