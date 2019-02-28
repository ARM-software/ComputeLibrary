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
#ifndef __ARM_COMPUTE_GRAPH_DETECTION_OUTPUT_LAYER_NODE_H__
#define __ARM_COMPUTE_GRAPH_DETECTION_OUTPUT_LAYER_NODE_H__

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** DetectionOutput Layer node */
class DetectionOutputLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] detection_info DetectionOutput Layer information
     */
    DetectionOutputLayerNode(DetectionOutputLayerInfo detection_info);
    /** DetectionOutput metadata accessor
     *
     * @return DetectionOutput Layer info
     */
    DetectionOutputLayerInfo detection_output_info() const;
    /** Computes detection output output descriptor
     *
     * @param[in] input_descriptor Input descriptor
     * @param[in] info             DetectionOutput operation attributes
     *
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor, const DetectionOutputLayerInfo &info);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    DetectionOutputLayerInfo _info;

    // Each detection contains a bounding box, given by its coordinates xmin, ymin, xmax, ymax, associated at the respective image, label and a confidence
    static const int detection_size = 7;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_DETECTION_OUTPUT_LAYER_NODE_H__ */
