/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/graph/nodes/EltwiseLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
EltwiseLayerNode::EltwiseLayerNode(EltwiseOperation op, QuantizationInfo out_quant_info, ConvertPolicy c_policy, RoundingPolicy r_policy)
    : _op(op), _out_quant_info(out_quant_info), _convert_policy(c_policy), _rounding_policy(r_policy)
{
    _input_edges.resize(2, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

EltwiseOperation EltwiseLayerNode::eltwise_operation() const
{
    return _op;
}

ConvertPolicy EltwiseLayerNode::convert_policy() const
{
    return _convert_policy;
}

RoundingPolicy EltwiseLayerNode::rounding_policy() const
{
    return _rounding_policy;
}

bool EltwiseLayerNode::forward_descriptors()
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

TensorDescriptor EltwiseLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx, _op, _convert_policy, _rounding_policy);

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    auto output_info = src->desc();

    if(!_out_quant_info.empty())
    {
        output_info.set_quantization_info(_out_quant_info);
    }

    return output_info;
}

NodeType EltwiseLayerNode::type() const
{
    return NodeType::EltwiseLayer;
}

void EltwiseLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
