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
#include "arm_compute/graph/nodes/ConvolutionLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
ConvolutionLayerNode::ConvolutionLayerNode(PadStrideInfo info, ConvolutionMethod method)
    : _info(std::move(info)), _method(method)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

void ConvolutionLayerNode::set_convolution_method(ConvolutionMethod method)
{
    _method = method;
}

ConvolutionMethod ConvolutionLayerNode::convolution_method() const
{
    return _method;
}

PadStrideInfo ConvolutionLayerNode::convolution_info() const
{
    return _info;
}

TensorShape ConvolutionLayerNode::compute_output_shape(TensorShape input_shape, TensorShape weights_shape, PadStrideInfo info)
{
    unsigned int output_width  = 0;
    unsigned int output_height = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input_shape.x(), input_shape.y(), weights_shape.x(), weights_shape.y(), info);

    TensorShape output_shape{ input_shape };
    output_shape.set(0, output_width);
    output_shape.set(1, output_height);
    output_shape.set(2, weights_shape[3]);

    return output_shape;
}

bool ConvolutionLayerNode::forward_descriptors()
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

TensorDescriptor ConvolutionLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    const Tensor *src     = input(0);
    const Tensor *weights = input(1);

    ARM_COMPUTE_ERROR_ON(src == nullptr || weights == nullptr);

    TensorDescriptor output_info  = src->desc();
    TensorShape      output_shape = compute_output_shape(src->desc().shape, weights->desc().shape, _info);
    output_info.shape             = output_shape;
    return output_info;
}

Status ConvolutionLayerNode::validate()
{
    return Status{};
}

NodeType ConvolutionLayerNode::type() const
{
    return NodeType::ConvolutionLayer;
}

void ConvolutionLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute