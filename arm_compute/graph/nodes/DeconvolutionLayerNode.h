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
#ifndef ARM_COMPUTE_GRAPH_DECONVOLUTION_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_DECONVOLUTION_LAYER_NODE_H

#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/LayerDescriptors.h"

namespace arm_compute
{
namespace graph
{
/** Deconvolution Layer node */
class DeconvolutionLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] descriptor Contains information used by this layer described in @ref descriptors::DeconvolutionLayerDescriptor
     */
    DeconvolutionLayerNode(const descriptors::DeconvolutionLayerDescriptor &descriptor);
    /** Deconvolution metadata accessor
     *
     * @return Deconvolution information
     */
    PadStrideInfo deconvolution_info() const;
    /** Computes deconvolution output descriptor
     *
     * @param[in] input_descriptor   Input descriptor
     * @param[in] weights_descriptor Weights descriptor
     * @param[in] info               Convolution operation attributes
     *
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                      const TensorDescriptor &weights_descriptor,
                                                      const PadStrideInfo    &info);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    descriptors::DeconvolutionLayerDescriptor descriptor;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_DECONVOLUTION_LAYER_NODE_H */
