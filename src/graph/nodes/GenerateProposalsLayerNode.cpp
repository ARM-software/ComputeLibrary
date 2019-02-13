/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/graph/nodes/GenerateProposalsLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace graph
{
GenerateProposalsLayerNode::GenerateProposalsLayerNode(GenerateProposalsInfo &info)
    : _info(info)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(3, NullTensorID);
}

const GenerateProposalsInfo &GenerateProposalsLayerNode::info() const
{
    return _info;
}

bool GenerateProposalsLayerNode::forward_descriptors()
{
    if((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (input_id(2) != NullTensorID) && (output_id(0) != NullTensorID) && (output_id(1) != NullTensorID)
       && (output_id(2) != NullTensorID))
    {
        for(unsigned int i = 0; i < 3; ++i)
        {
            Tensor *dst = output(i);
            ARM_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(i);
        }
        return true;
    }
    return false;
}

TensorDescriptor GenerateProposalsLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(idx > 3);

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);
    TensorDescriptor output_desc = src->desc();

    switch(idx)
    {
        case 0:
            // Configure proposals output
            output_desc.shape = TensorShape(5, src->desc().shape.total_size());
            break;
        case 1:
            // Configure scores_out output
            output_desc.shape = TensorShape(src->desc().shape.total_size());
            break;
        case 2:
            // Configure num_valid_proposals
            output_desc.shape     = TensorShape(1);
            output_desc.data_type = DataType::U32;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported output index");
    }
    return output_desc;
}

NodeType GenerateProposalsLayerNode::type() const
{
    return NodeType::GenerateProposalsLayer;
}

void GenerateProposalsLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
