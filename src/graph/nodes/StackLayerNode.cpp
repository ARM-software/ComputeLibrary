/*
 * Copyright (c) 2019 Arm Limited.
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
#include "arm_compute/graph/nodes/StackLayerNode.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace graph
{
StackLayerNode::StackLayerNode(unsigned int total_nodes, int axis)
    : _total_nodes(total_nodes), _axis(axis)
{
    _input_edges.resize(_total_nodes, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

int StackLayerNode::axis() const
{
    return _axis;
}

TensorDescriptor StackLayerNode::compute_output_descriptor(const std::vector<TensorDescriptor> &input_descriptors,
                                                           int                                  axis)
{
    ARM_COMPUTE_ERROR_ON(input_descriptors.size() == 0);

    TensorDescriptor output_descriptor = input_descriptors[0];

    const TensorInfo   input_info(input_descriptors[0].shape, 1, input_descriptors[0].data_type);
    const unsigned int num_tensors = input_descriptors.size();

    output_descriptor.shape = arm_compute::misc::shape_calculator::compute_stack_shape(input_info, axis, num_tensors);

    return output_descriptor;
}

bool StackLayerNode::forward_descriptors()
{
    if(_outputs[0] != NullTensorID)
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor StackLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    // Check if all input tensors are set
    bool are_all_inputs_set = std::all_of(std::begin(_input_edges), std::end(_input_edges), [](const EdgeID & eid)
    {
        return eid != EmptyEdgeID;
    });

    TensorDescriptor output_info = {};

    if(are_all_inputs_set)
    {
        std::vector<TensorDescriptor> inputs_descriptors;
        for(unsigned int i = 0; i < _input_edges.size(); ++i)
        {
            const Tensor *t = _graph->tensor(input_id(i));
            ARM_COMPUTE_ERROR_ON(t == nullptr);
            inputs_descriptors.push_back(t->desc());
        }
        output_info = compute_output_descriptor(inputs_descriptors, _axis);
    }

    return output_info;
}

NodeType StackLayerNode::type() const
{
    return NodeType::StackLayer;
}

void StackLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
