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
#include "arm_compute/graph2/nodes/SplitLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/INodeVisitor.h"

namespace arm_compute
{
namespace graph2
{
SplitLayerNode::SplitLayerNode(unsigned int num_splits, unsigned int axis)
    : _num_splits(num_splits), _axis(axis)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(num_splits, NullTensorID);
}

unsigned int SplitLayerNode::num_splits() const
{
    return _num_splits;
}

unsigned int SplitLayerNode::axis() const
{
    return _axis;
}

std::pair<TensorShape, Coordinates> SplitLayerNode::compute_output_shape(TensorShape input_shape, unsigned int num_splits, unsigned int axis, unsigned int idx)
{
    ARM_COMPUTE_ERROR_ON(axis >= input_shape.num_dimensions());
    ARM_COMPUTE_ERROR_ON_MSG(input_shape[axis] % num_splits, "Split should be exact");

    const unsigned int split_size = input_shape[axis] / num_splits;

    TensorShape output_shape = input_shape;
    output_shape.set(axis, split_size);

    Coordinates coords;
    coords.set(axis, idx * split_size);

    return std::make_pair(output_shape, coords);
}

bool SplitLayerNode::forward_descriptors()
{
    if(input_id(0) != NullTensorID)
    {
        for(unsigned int i = 0; i < _outputs.size(); ++i)
        {
            if(output_id(i) != NullTensorID)
            {
                Tensor *dst_i = output(i);
                ARM_COMPUTE_ERROR_ON(dst_i == nullptr);
                dst_i->desc() = configure_output(i);
            }
        }
        return true;
    }
    return false;
}

TensorDescriptor SplitLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorShape output_shape;

    TensorDescriptor output_info = src->desc();
    std::tie(output_shape, std::ignore) = compute_output_shape(src->desc().shape, _num_splits, _axis, idx);
    output_info.shape = output_shape;

    return output_info;
}

Status SplitLayerNode::validate()
{
    return Status{};
}

NodeType SplitLayerNode::type() const
{
    return NodeType::SplitLayer;
}

void SplitLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph2
} // namespace arm_compute