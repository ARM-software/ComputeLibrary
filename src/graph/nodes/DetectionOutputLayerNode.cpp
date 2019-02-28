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
#include "arm_compute/graph/nodes/DetectionOutputLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
DetectionOutputLayerNode::DetectionOutputLayerNode(DetectionOutputLayerInfo detection_info)
    : _info(detection_info)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

DetectionOutputLayerInfo DetectionOutputLayerNode::detection_output_info() const
{
    return _info;
}

TensorDescriptor DetectionOutputLayerNode::compute_output_descriptor(const TensorDescriptor         &input_descriptor,
                                                                     const DetectionOutputLayerInfo &info)
{
    const unsigned int max_size = info.keep_top_k() * ((input_descriptor.shape.num_dimensions() > 1) ? input_descriptor.shape[1] : 1);

    TensorDescriptor output_descriptor = input_descriptor;
    output_descriptor.shape.set(0, detection_size);
    output_descriptor.shape.set(1, max_size);

    return output_descriptor;
}

bool DetectionOutputLayerNode::forward_descriptors()
{
    if((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (input_id(2) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor DetectionOutputLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *input0 = input(0);
    ARM_COMPUTE_ERROR_ON(input0 == nullptr);

    return compute_output_descriptor(input0->desc(), _info);
}

NodeType DetectionOutputLayerNode::type() const
{
    return NodeType::DetectionOutputLayer;
}

void DetectionOutputLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
