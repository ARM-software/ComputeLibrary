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
#include "arm_compute/graph/nodes/FusedConvolutionBatchNormalizationNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
FusedConvolutionBatchNormalizationNode::FusedConvolutionBatchNormalizationNode(float epsilon, PadStrideInfo info,
                                                                               unsigned int      num_groups,
                                                                               ConvolutionMethod method,
                                                                               FastMathHint      fast_math_hint,
                                                                               QuantizationInfo out_quant_info, ActivationLayerInfo fused_activation)
    : _epsilon(epsilon), _info(std::move(info)), _num_groups(num_groups), _method(method), _fast_math_hint(fast_math_hint), _out_quant_info(out_quant_info), _fused_activation(fused_activation)
{
    _input_edges.resize(7, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

void FusedConvolutionBatchNormalizationNode::set_convolution_method(ConvolutionMethod method)
{
    _method = method;
}

float FusedConvolutionBatchNormalizationNode::epsilon() const
{
    return _epsilon;
}

ConvolutionMethod FusedConvolutionBatchNormalizationNode::convolution_method() const
{
    return _method;
}

void FusedConvolutionBatchNormalizationNode::set_fast_math_hint(FastMathHint hint)
{
    _fast_math_hint = hint;
}

FastMathHint FusedConvolutionBatchNormalizationNode::fast_math_hint() const
{
    return _fast_math_hint;
}

PadStrideInfo FusedConvolutionBatchNormalizationNode::convolution_info() const
{
    return _info;
}

unsigned int FusedConvolutionBatchNormalizationNode::num_groups() const
{
    return _num_groups;
}

ActivationLayerInfo FusedConvolutionBatchNormalizationNode::fused_activation() const
{
    return _fused_activation;
}

void FusedConvolutionBatchNormalizationNode::set_fused_activation(ActivationLayerInfo fused_activation)
{
    _fused_activation = fused_activation;
}

TensorDescriptor FusedConvolutionBatchNormalizationNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                                   const TensorDescriptor &weights_descriptor,
                                                                                   const PadStrideInfo    &info)
{
    unsigned int output_width  = 0;
    unsigned int output_height = 0;

    const unsigned int input_width   = get_dimension_size(input_descriptor, DataLayoutDimension::WIDTH);
    const unsigned int input_height  = get_dimension_size(input_descriptor, DataLayoutDimension::HEIGHT);
    const unsigned int kernel_width  = get_dimension_size(weights_descriptor, DataLayoutDimension::WIDTH);
    const unsigned int kernel_height = get_dimension_size(weights_descriptor, DataLayoutDimension::HEIGHT);

    std::tie(output_width, output_height) = scaled_dimensions(input_width, input_height, kernel_width, kernel_height, info);

    const DataLayout data_layout       = input_descriptor.layout;
    TensorDescriptor output_descriptor = input_descriptor;
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::WIDTH), output_width);
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::HEIGHT), output_height);
    output_descriptor.shape.set(get_dimension_idx(data_layout, DataLayoutDimension::CHANNEL), weights_descriptor.shape[3]);

    return output_descriptor;
}

bool FusedConvolutionBatchNormalizationNode::forward_descriptors()
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

TensorDescriptor FusedConvolutionBatchNormalizationNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    const Tensor *src     = input(0);
    const Tensor *weights = input(1);

    ARM_COMPUTE_ERROR_ON(src == nullptr || weights == nullptr);

    TensorDescriptor output_info = compute_output_descriptor(src->desc(), weights->desc(), _info);
    if(!_out_quant_info.empty())
    {
        output_info.quant_info = _out_quant_info;
    }

    return output_info;
}

NodeType FusedConvolutionBatchNormalizationNode::type() const
{
    return FusedConvolutionBatchNormalizationNode::node_type;
}

void FusedConvolutionBatchNormalizationNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
