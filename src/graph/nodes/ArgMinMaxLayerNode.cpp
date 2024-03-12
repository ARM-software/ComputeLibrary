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
#include "arm_compute/graph/nodes/ArgMinMaxLayerNode.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
ArgMinMaxLayerNode::ArgMinMaxLayerNode(ReductionOperation op,
                                       unsigned int       axis,
                                       DataType           out_data_type,
                                       QuantizationInfo   out_quant_info)
    : _op(op), _axis(axis), _out_data_type(out_data_type), _out_quant_info(std::move(out_quant_info))
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

ReductionOperation ArgMinMaxLayerNode::reduction_operation() const
{
    return _op;
}

unsigned int ArgMinMaxLayerNode::axis() const
{
    return _axis;
}

DataType ArgMinMaxLayerNode::out_data_type() const
{
    return _out_data_type;
}

bool ArgMinMaxLayerNode::forward_descriptors()
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

TensorDescriptor ArgMinMaxLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_info = src->desc();
    if (!_out_quant_info.empty())
    {
        output_info.quant_info = _out_quant_info;
    }

    if (_out_data_type != DataType::UNKNOWN)
    {
        output_info.data_type = _out_data_type;
    }

    TensorShape output_shape =
        arm_compute::misc::shape_calculator::compute_reduced_shape(output_info.shape, _axis, false);
    output_info.set_shape(output_shape);

    return output_info;
}

NodeType ArgMinMaxLayerNode::type() const
{
    return ArgMinMaxLayerNode::node_type;
}

void ArgMinMaxLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
