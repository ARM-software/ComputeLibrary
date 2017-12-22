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
#include "arm_compute/graph2/nodes/FullyConnectedLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/INodeVisitor.h"

namespace arm_compute
{
namespace graph2
{
FullyConnectedLayerNode::FullyConnectedLayerNode(unsigned int num_outputs)
    : _num_outputs(num_outputs)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

TensorShape FullyConnectedLayerNode::compute_weights_shape(TensorShape input_shape, unsigned int num_outputs)
{
    unsigned int num_weights    = 1;
    unsigned int num_dimensions = input_shape.num_dimensions();
    // Ignore the batch dimension if there is one:
    if(num_dimensions == 2 || num_dimensions == 4)
    {
        num_dimensions--;
    }
    for(unsigned int i = 0; i < num_dimensions; i++)
    {
        num_weights *= input_shape[i];
    }
    return TensorShape(num_weights, num_outputs);
}

TensorShape FullyConnectedLayerNode::compute_output_shape(TensorShape input_shape, unsigned int num_outputs)
{
    // Note: Only 1D batch space is supported at the moment
    unsigned int batches = input_shape[1];
    if(input_shape.num_dimensions() > 2)
    {
        batches = input_shape[3];
    }
    return TensorShape(num_outputs, batches);
}

bool FullyConnectedLayerNode::forward_descriptors()
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

TensorDescriptor FullyConnectedLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_info  = src->desc();
    TensorShape      output_shape = compute_output_shape(src->desc().shape, _num_outputs);
    output_info.shape             = output_shape;
    return output_info;
}

Status FullyConnectedLayerNode::validate()
{
    return Status{};
}

NodeType FullyConnectedLayerNode::type() const
{
    return NodeType::FullyConnectedLayer;
}

void FullyConnectedLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph2
} // namespace arm_compute