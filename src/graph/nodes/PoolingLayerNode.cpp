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
#include "arm_compute/graph/nodes/PoolingLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

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

TensorDescriptor PoolingLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                             PoolingLayerInfo        info)
{
    unsigned int pooled_width  = 0;
    unsigned int pooled_height = 0;

    const unsigned int input_width  = get_dimension_size(input_descriptor, DataLayoutDimension::WIDTH);
    const unsigned int input_height = get_dimension_size(input_descriptor, DataLayoutDimension::HEIGHT);
    const unsigned int pool_size_x  = info.is_global_pooling() ? input_width : info.pool_size().width;
    const unsigned int pool_size_y  = info.is_global_pooling() ? input_height : info.pool_size().height;

    std::tie(pooled_width, pooled_height) = scaled_dimensions(input_width, input_height, pool_size_x, pool_size_y, info.pad_stride_info());

    const DataLayout data_layout       = input_descriptor.layout;
    TensorDescriptor output_descriptor = input_descriptor;
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::WIDTH), pooled_width);
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::HEIGHT), pooled_height);

    return output_descriptor;
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

    return compute_output_descriptor(src->desc(), _info);
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