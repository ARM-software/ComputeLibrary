/*
 * Copyright (c) 2018 Arm Limited.
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

#include "arm_compute/graph/nodes/ROIAlignLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace graph
{
ROIAlignLayerNode::ROIAlignLayerNode(ROIPoolingLayerInfo &pool_info)
    : _pool_info(pool_info)
{
    _input_edges.resize(2, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

const ROIPoolingLayerInfo &ROIAlignLayerNode::pooling_info() const
{
    return _pool_info;
}

bool ROIAlignLayerNode::forward_descriptors()
{
    if((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor ROIAlignLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src  = input(0);
    const Tensor *rois = input(1);
    ARM_COMPUTE_ERROR_ON(src == nullptr);
    ARM_COMPUTE_ERROR_ON(rois == nullptr);

    TensorDescriptor output_desc = src->desc();

    const size_t idx_n = get_data_layout_dimension_index(output_desc.layout, DataLayoutDimension::BATCHES);
    const size_t idx_c = get_data_layout_dimension_index(output_desc.layout, DataLayoutDimension::CHANNEL);
    const size_t idx_h = get_data_layout_dimension_index(output_desc.layout, DataLayoutDimension::HEIGHT);
    const size_t idx_w = get_data_layout_dimension_index(output_desc.layout, DataLayoutDimension::WIDTH);

    output_desc.shape.set(idx_n, rois->desc().shape[1]);
    output_desc.shape.set(idx_c, src->desc().shape[idx_c]);
    output_desc.shape.set(idx_h, _pool_info.pooled_height());
    output_desc.shape.set(idx_w, _pool_info.pooled_width());

    return output_desc;
}

NodeType ROIAlignLayerNode::type() const
{
    return NodeType::ROIAlignLayer;
}

void ROIAlignLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute