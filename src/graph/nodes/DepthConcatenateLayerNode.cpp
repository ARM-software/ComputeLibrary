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
    _input_edges.resize(total_nodes, EmptyEdgeID);
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

TensorShape DepthConcatenateLayerNode::compute_output_shape(const std::vector<TensorShape> &input_shapes)
{
    ARM_COMPUTE_ERROR_ON(input_shapes.size() == 0);

    TensorShape output_shape = input_shapes[0];

    size_t max_x = 0;
    size_t max_y = 0;
    size_t depth = 0;

    for(const auto &shape : input_shapes)
    {
        max_x = std::max(shape.x(), max_x);
        max_y = std::max(shape.y(), max_y);
        depth += shape.z();
    }

    output_shape.set(0, max_x);
    output_shape.set(1, max_y);
    output_shape.set(2, depth);

    return output_shape;
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
        std::vector<TensorShape> inputs_shapes;
        for(unsigned int i = 0; i < _input_edges.size(); ++i)
        {
            const Tensor *t = _graph->tensor(input_id(i));
            ARM_COMPUTE_ERROR_ON(t == nullptr);
            inputs_shapes.push_back(t->desc().shape);
        }
        output_info              = input(0)->desc();
        TensorShape output_shape = compute_output_shape(inputs_shapes);
        output_info.shape        = output_shape;
    }

    return output_info;
}

Status DepthConcatenateLayerNode::validate()
{
    ARM_COMPUTE_UNUSED(_total_nodes);
    return Status{};
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