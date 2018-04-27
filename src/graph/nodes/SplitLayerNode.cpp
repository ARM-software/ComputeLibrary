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
#include "arm_compute/graph/nodes/SplitLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
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

std::pair<TensorDescriptor, Coordinates> SplitLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                                   unsigned int num_splits, unsigned int axis, unsigned int idx)
{
    const unsigned int split_size = input_descriptor.shape[axis] / num_splits;

    TensorDescriptor output_descriptor = input_descriptor;
    output_descriptor.shape.set(axis, split_size);

    Coordinates coords;
    coords.set(axis, idx * split_size);

    return std::make_pair(output_descriptor, coords);
}

bool SplitLayerNode::forward_descriptors()
{
    if(input_id(0) != NullTensorID)
    {
        validate();
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

    TensorDescriptor output_info;
    std::tie(output_info, std::ignore) = compute_output_descriptor(src->desc(), _num_splits, _axis, idx);

    return output_info;
}

Status SplitLayerNode::validate() const
{
    const Tensor *src = input(0);
    ARM_COMPUTE_RETURN_ERROR_ON(src == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(_axis >= src->desc().shape.num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->desc().shape[_axis] % _num_splits, "Split should be exact");

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
} // namespace graph
} // namespace arm_compute