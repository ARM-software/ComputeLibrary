/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/graph/nodes/PadLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace graph
{
PadLayerNode::PadLayerNode(const PaddingList &padding, PixelValue pad_value)
    : _padding(padding), _pad_value(pad_value)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

const PaddingList &PadLayerNode::padding() const
{
    return _padding;
}

PixelValue PadLayerNode::pad_value() const
{
    return _pad_value;
}

bool PadLayerNode::forward_descriptors()
{
    if((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor PadLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor  output_desc = src->desc();
    const TensorShape input_shape = src->desc().shape;
    for(size_t dim = 0; dim < _padding.size(); ++dim)
    {
        output_desc.shape.set(dim, _padding[dim].first + input_shape[dim] + _padding[dim].second);
    }

    return output_desc;
}

NodeType PadLayerNode::type() const
{
    return NodeType::PadLayer;
}

void PadLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
