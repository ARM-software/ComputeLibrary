/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/graph/nodes/StridedSliceLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
StridedSliceLayerNode::StridedSliceLayerNode(const Coordinates    &starts,
                                             const Coordinates    &ends,
                                             const BiStrides      &strides,
                                             StridedSliceLayerInfo info)
    : _starts(starts), _ends(ends), _strides(strides), _info(std::move(info))
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

Coordinates StridedSliceLayerNode::starts() const
{
    return _starts;
}

Coordinates StridedSliceLayerNode::ends() const
{
    return _ends;
}

BiStrides StridedSliceLayerNode::strides() const
{
    return _strides;
}

StridedSliceLayerInfo StridedSliceLayerNode::strided_slice_info() const
{
    return _info;
}

TensorDescriptor StridedSliceLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                  const Coordinates      &starts,
                                                                  const Coordinates      &ends,
                                                                  const BiStrides        &strides,
                                                                  StridedSliceLayerInfo   info)
{
    using namespace arm_compute::helpers::tensor_transform;

    TensorDescriptor output_desc = input_descriptor;
    output_desc.shape            = compute_strided_slice_output_shape(input_descriptor.shape, starts, ends, strides,
                                                                      info.begin_mask(), info.end_mask(), info.shrink_axis_mask());

    return output_desc;
}

bool StridedSliceLayerNode::forward_descriptors()
{
    if ((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor StridedSliceLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    return compute_output_descriptor(src->desc(), _starts, _ends, _strides, _info);
}

NodeType StridedSliceLayerNode::type() const
{
    return NodeType::StridedSliceLayer;
}

void StridedSliceLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
