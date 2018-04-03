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
#include "arm_compute/graph/nodes/PoolingLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
PoolingLayerNode::PoolingLayerNode(PoolingLayerInfo pool_info)
    : _info(std::move(pool_info))
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

PoolingLayerInfo PoolingLayerNode::pooling_info() const
{
    return _info;
}

TensorShape PoolingLayerNode::compute_output_shape(TensorShape input_shape, PoolingLayerInfo info)
{
    const int pool_size_x = info.is_global_pooling() ? input_shape.x() : info.pool_size().width;
    const int pool_size_y = info.is_global_pooling() ? input_shape.y() : info.pool_size().height;

    unsigned int pooled_width  = 0;
    unsigned int pooled_height = 0;
    std::tie(pooled_width, pooled_height) = scaled_dimensions(input_shape.x(), input_shape.y(), pool_size_x, pool_size_y, info.pad_stride_info());

    TensorShape output_shape{ input_shape };
    output_shape.set(0, pooled_width);
    output_shape.set(1, pooled_height);

    return output_shape;
}

bool PoolingLayerNode::forward_descriptors()
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

TensorDescriptor PoolingLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_info  = src->desc();
    TensorShape      output_shape = compute_output_shape(src->desc().shape, _info);
    output_info.shape             = output_shape;
    return output_info;
}

Status PoolingLayerNode::validate()
{
    return Status{};
}

NodeType PoolingLayerNode::type() const
{
    return NodeType::PoolingLayer;
}

void PoolingLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute