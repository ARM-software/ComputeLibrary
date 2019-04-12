/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/graph/nodes/ResizeLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
ResizeLayerNode::ResizeLayerNode(InterpolationPolicy policy, float scale_width, float scale_height)
    : _policy(policy), _scale_width(scale_width), _scale_height(scale_height)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

InterpolationPolicy ResizeLayerNode::policy() const
{
    return _policy;
}

std::pair<float, float> ResizeLayerNode::scaling_factor() const
{
    return std::make_pair(_scale_width, _scale_height);
}

bool ResizeLayerNode::forward_descriptors()
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

TensorDescriptor ResizeLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    const DataLayout data_layout = src->desc().layout;
    TensorDescriptor output_desc = src->desc();
    size_t           width_idx   = get_dimension_idx(data_layout, DataLayoutDimension::WIDTH);
    size_t           height_idx  = get_dimension_idx(data_layout, DataLayoutDimension::HEIGHT);
    output_desc.shape.set(width_idx, static_cast<int>(output_desc.shape[width_idx] * _scale_width));
    output_desc.shape.set(height_idx, static_cast<int>(output_desc.shape[height_idx] * _scale_height));

    return output_desc;
}

NodeType ResizeLayerNode::type() const
{
    return NodeType::ResizeLayer;
}

void ResizeLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute