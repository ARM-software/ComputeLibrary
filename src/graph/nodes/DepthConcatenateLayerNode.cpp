/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/graph/nodes/DepthConcatenateLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
DepthConcatenateLayerNode::DepthConcatenateLayerNode(unsigned int total_nodes)
    : _total_nodes(total_nodes), _is_enabled(true)
{
    _input_edges.resize(_total_nodes, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

void DepthConcatenateLayerNode::set_enabled(bool is_enabled)
{
    _is_enabled = is_enabled;
}

bool DepthConcatenateLayerNode::is_enabled() const
{
    return _is_enabled;
}

TensorDescriptor DepthConcatenateLayerNode::compute_output_descriptor(const std::vector<TensorDescriptor> &input_descriptors)
{
    ARM_COMPUTE_ERROR_ON(input_descriptors.size() == 0);

    TensorDescriptor output_descriptor = input_descriptors[0];

    size_t max_x = 0;
    size_t max_y = 0;
    size_t depth = 0;

    for(const auto &input_descriptor : input_descriptors)
    {
        max_x = std::max(input_descriptor.shape.x(), max_x);
        max_y = std::max(input_descriptor.shape.y(), max_y);
        depth += input_descriptor.shape.z();
    }

    output_descriptor.shape.set(0, max_x);
    output_descriptor.shape.set(1, max_y);
    output_descriptor.shape.set(2, depth);

    return output_descriptor;
}

bool DepthConcatenateLayerNode::forward_descriptors()
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

TensorDescriptor DepthConcatenateLayerNode::configure_output(size_t idx) const
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
        output_info = compute_output_descriptor(inputs_descriptors);
    }

    return output_info;
}

NodeType DepthConcatenateLayerNode::type() const
{
    return NodeType::DepthConcatenateLayer;
}

void DepthConcatenateLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute