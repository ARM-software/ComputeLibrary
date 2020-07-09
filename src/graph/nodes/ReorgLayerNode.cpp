/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#include "arm_compute/graph/nodes/ReorgLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
ReorgLayerNode::ReorgLayerNode(int stride)
    : _stride(stride)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

int ReorgLayerNode::stride() const
{
    return _stride;
}

TensorDescriptor ReorgLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor, int stride)
{
    const unsigned int input_width   = get_dimension_size(input_descriptor, DataLayoutDimension::WIDTH);
    const unsigned int input_height  = get_dimension_size(input_descriptor, DataLayoutDimension::HEIGHT);
    const unsigned int input_channel = get_dimension_size(input_descriptor, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_ERROR_ON(stride <= 0);
    ARM_COMPUTE_ERROR_ON_MSG((input_width % stride != 0), "The width of the input tensor must be a multiple of stride");
    ARM_COMPUTE_ERROR_ON_MSG((input_height % stride != 0), "The height of the input tensor must be a multiple of stride");

    const DataLayout data_layout       = input_descriptor.layout;
    TensorDescriptor output_descriptor = input_descriptor;
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::WIDTH), input_width / stride);
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::HEIGHT), input_height / stride);
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::CHANNEL), input_channel * stride * stride);

    return output_descriptor;
}

bool ReorgLayerNode::forward_descriptors()
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

TensorDescriptor ReorgLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    return compute_output_descriptor(src->desc(), _stride);
}

NodeType ReorgLayerNode::type() const
{
    return NodeType::ReorgLayer;
}

void ReorgLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute