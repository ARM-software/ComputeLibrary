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
#include "arm_compute/graph2/nodes/NormalizationLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/INodeVisitor.h"

namespace arm_compute
{
namespace graph2
{
NormalizationLayerNode::NormalizationLayerNode(NormalizationLayerInfo norm_info)
    : _info(norm_info)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

NormalizationLayerInfo NormalizationLayerNode::normalization_info() const
{
    return _info;
}

bool NormalizationLayerNode::forward_descriptors()
{
    if(input_id(0) != NullTensorID && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor NormalizationLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    return src->desc();
}

Status NormalizationLayerNode::validate()
{
    return Status{};
}

NodeType NormalizationLayerNode::type() const
{
    return NodeType::NormalizationLayer;
}

void NormalizationLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph2
} // namespace arm_compute