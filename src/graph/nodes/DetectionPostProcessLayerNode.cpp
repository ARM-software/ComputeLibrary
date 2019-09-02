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
#include "arm_compute/graph/nodes/DetectionPostProcessLayerNode.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
DetectionPostProcessLayerNode::DetectionPostProcessLayerNode(DetectionPostProcessLayerInfo detection_info)
    : _info(detection_info)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(4, NullTensorID);
}

DetectionPostProcessLayerInfo DetectionPostProcessLayerNode::detection_post_process_info() const
{
    return _info;
}

bool DetectionPostProcessLayerNode::forward_descriptors()
{
    if((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (input_id(2) != NullTensorID) && (output_id(0) != NullTensorID) && (output_id(1) != NullTensorID)
       && (output_id(2) != NullTensorID) && (output_id(3) != NullTensorID))
    {
        for(unsigned int i = 0; i < 4; ++i)
        {
            Tensor *dst = output(i);
            ARM_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(i);
        }
        return true;
    }
    return false;
}

TensorDescriptor DetectionPostProcessLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    TensorDescriptor   output_desc;
    const unsigned int num_detected_box = _info.max_detections() * _info.max_classes_per_detection();

    switch(idx)
    {
        case 0:
            // Configure boxes output
            output_desc.shape = TensorShape(kNumCoordBox, num_detected_box, kBatchSize);
            break;
        case 1:
        case 2:
            // Configure classes or scores output
            output_desc.shape = TensorShape(num_detected_box, kBatchSize);
            break;
        case 3:
            // Configure num_detection
            output_desc.shape = TensorShape(1);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported output index");
    }
    output_desc.data_type = DataType::F32;

    return output_desc;
}

NodeType DetectionPostProcessLayerNode::type() const
{
    return NodeType::DetectionPostProcessLayer;
}

void DetectionPostProcessLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute