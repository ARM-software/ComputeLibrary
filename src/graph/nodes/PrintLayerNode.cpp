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
#include "arm_compute/graph/nodes/PrintLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
PrintLayerNode::PrintLayerNode(std::ostream &stream, const IOFormatInfo &format_info, const std::function<ITensor *(ITensor *)> transform)
    : _stream(stream), _format_info(format_info), _transform(transform)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

std::ostream &PrintLayerNode::stream() const
{
    return _stream;
}

const IOFormatInfo PrintLayerNode::format_info() const
{
    return _format_info;
}

const std::function<ITensor *(ITensor *)> PrintLayerNode::transform() const
{
    return _transform;
}

bool PrintLayerNode::forward_descriptors()
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

TensorDescriptor PrintLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_desc = src->desc();
    return output_desc;
}

NodeType PrintLayerNode::type() const
{
    return NodeType::PrintLayer;
}

void PrintLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute