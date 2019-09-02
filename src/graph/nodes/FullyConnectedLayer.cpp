/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/graph/nodes/FullyConnectedLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
FullyConnectedLayerNode::FullyConnectedLayerNode(unsigned int num_outputs, QuantizationInfo out_quant_info, FullyConnectedLayerInfo fc_info)
    : _num_outputs(num_outputs), _out_quant_info(std::move(out_quant_info)), _info(fc_info)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

TensorDescriptor FullyConnectedLayerNode::compute_weights_descriptor(const TensorDescriptor &input_descriptor,
                                                                     unsigned int            num_outputs,
                                                                     FullyConnectedLayerInfo fc_info,
                                                                     const QuantizationInfo &weights_quant_info)
{
    unsigned int num_weights    = 1;
    unsigned int num_dimensions = input_descriptor.shape.num_dimensions();
    // Ignore the batch dimension if there is one:
    if(num_dimensions == 2 || num_dimensions == 4)
    {
        num_dimensions--;
    }
    for(unsigned int i = 0; i < num_dimensions; i++)
    {
        num_weights *= input_descriptor.shape[i];
    }

    TensorDescriptor weights_descriptor = input_descriptor;
    weights_descriptor.shape            = TensorShape(num_weights, num_outputs);

    // If weights are tranposed, use tranposed shape
    if(!fc_info.transpose_weights)
    {
        weights_descriptor.shape = TensorShape(num_outputs, num_weights);
    }

    // Set quantization info if present
    if(!weights_quant_info.empty())
    {
        weights_descriptor.quant_info = weights_quant_info;
    }

    return weights_descriptor;
}

TensorDescriptor FullyConnectedLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                    unsigned int            num_outputs,
                                                                    const QuantizationInfo &out_quant_info)
{
    // Note: Only 1D batch space is supported at the moment
    unsigned int batches = input_descriptor.shape[1];
    if(input_descriptor.shape.num_dimensions() > 2)
    {
        batches = input_descriptor.shape[3];
    }

    // Set descriptor shape
    TensorDescriptor output_descriptor = input_descriptor;
    output_descriptor.shape            = TensorShape(num_outputs, batches);

    // Set quantization info if present
    if(!out_quant_info.empty())
    {
        output_descriptor.quant_info = out_quant_info;
    }

    return output_descriptor;
}

FullyConnectedLayerInfo FullyConnectedLayerNode::info() const
{
    return _info;
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

    return compute_output_descriptor(src->desc(), _num_outputs, _out_quant_info);
}

NodeType FullyConnectedLayerNode::type() const
{
    return NodeType::FullyConnectedLayer;
}

void FullyConnectedLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute