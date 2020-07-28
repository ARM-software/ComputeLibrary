/*
 * Copyright (c) 2018-2020 Arm Limited.
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

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
EltwiseLayerNode::EltwiseLayerNode(const descriptors::EltwiseLayerDescriptor &descriptor)
    : descriptor(descriptor)
{
    _input_edges.resize(2, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

EltwiseOperation EltwiseLayerNode::eltwise_operation() const
{
    return descriptor.op;
}

ConvertPolicy EltwiseLayerNode::convert_policy() const
{
    return descriptor.c_policy;
}

RoundingPolicy EltwiseLayerNode::rounding_policy() const
{
    return descriptor.r_policy;
}

ActivationLayerInfo EltwiseLayerNode::fused_activation() const
{
    return descriptor.fused_activation;
}

QuantizationInfo EltwiseLayerNode::output_quant_info() const
{
    return descriptor.out_quant_info;
}

void EltwiseLayerNode::set_fused_activation(ActivationLayerInfo fused_activation)
{
    descriptor.fused_activation = fused_activation;
}

bool EltwiseLayerNode::forward_descriptors()
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

TensorDescriptor EltwiseLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);

    const Tensor *src1 = input(0);
    ARM_COMPUTE_ERROR_ON(src1 == nullptr);

    const Tensor *src2 = input(1);
    ARM_COMPUTE_ERROR_ON(src2 == nullptr);

    auto output_info = src1->desc();

    TensorShape out_shape = TensorShape::broadcast_shape(src1->desc().shape, src2->desc().shape);
    ARM_COMPUTE_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    output_info.set_shape(out_shape);

    if(!descriptor.out_quant_info.empty())
    {
        output_info.set_quantization_info(descriptor.out_quant_info);
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

UnaryEltwiseLayerNode::UnaryEltwiseLayerNode(const descriptors::UnaryEltwiseLayerDescriptor &descriptor)
    : descriptor(descriptor)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

descriptors::UnaryEltwiseLayerDescriptor UnaryEltwiseLayerNode::eltwise_descriptor() const
{
    return descriptor;
}

void UnaryEltwiseLayerNode::set_fused_activation(ActivationLayerInfo fused_activation)
{
    descriptor.fused_activation = fused_activation;
}

bool UnaryEltwiseLayerNode::forward_descriptors()
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

TensorDescriptor UnaryEltwiseLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    auto output_info = src->desc();

    if(!descriptor.out_quant_info.empty())
    {
        output_info.set_quantization_info(descriptor.out_quant_info);
    }

    return output_info;
}

NodeType UnaryEltwiseLayerNode::type() const
{
    return NodeType::UnaryEltwiseLayer;
}

void UnaryEltwiseLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}

} // namespace graph
} // namespace arm_compute
