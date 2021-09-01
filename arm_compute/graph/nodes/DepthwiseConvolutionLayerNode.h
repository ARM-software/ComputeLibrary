/*
 * Copyright (c) 2018-2019, 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_DEPTHWISE_CONVOLUTION_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_DEPTHWISE_CONVOLUTION_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Depthwise Convolution Layer node */
class DepthwiseConvolutionLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] info             Convolution layer attributes
     * @param[in] depth_multiplier (Optional) Depth multiplier parameter.
     * @param[in] method           (Optional) Depthwise convolution method to use
     * @param[in] out_quant_info   (Optional) Output quantization info
     */
    DepthwiseConvolutionLayerNode(PadStrideInfo              info,
                                  int                        depth_multiplier = 1,
                                  DepthwiseConvolutionMethod method           = DepthwiseConvolutionMethod::Default,
                                  QuantizationInfo           out_quant_info   = QuantizationInfo());
    /** Sets the depthwise convolution method to use
     *
     * @param[in] method Depthwise convolution method to use
     */
    void set_depthwise_convolution_method(DepthwiseConvolutionMethod method);
    /** Depthwise convolution layer method accessor
     *
     * @note This is an indication on which depthwise implementation to use,
     *       if it fails to be created the generic approach will be used
     *
     * @return Depthwise convolution layer method do be used by the node
     */
    DepthwiseConvolutionMethod depthwise_convolution_method() const;
    /** Depth multiplier accessor
     *
     * @return Depth multiplier
     */
    int depth_multiplier() const;
    /** Convolution metadata accessor
     *
     * @return Convolution information
     */
    PadStrideInfo convolution_info() const;
    /** Returns fused activation
     *
     * @return Fused activation
     */
    ActivationLayerInfo fused_activation() const;
    /** Sets fused activation
     *
     * @param[in] fused_activation Fused activation to set
     */
    void set_fused_activation(ActivationLayerInfo fused_activation);
    /** Sets convolution info
     *
     * @param[in] info Convolution info to set
     */
    void set_convolution_info(PadStrideInfo info);
    /** Computes depthwise convolution output descriptor
     *
     * @param[in] input_descriptor   Input descriptor
     * @param[in] weights_descriptor Weights descriptor
     * @param[in] info               Convolution operation attributes
     * @param[in] depth_multiplier   (Optional) Depth multiplier parameter.
     *
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                      const TensorDescriptor &weights_descriptor,
                                                      const PadStrideInfo    &info,
                                                      int                     depth_multiplier = 1);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

public:
    static constexpr NodeType node_type = NodeType::DepthwiseConvolutionLayer;

private:
    PadStrideInfo              _info;
    int                        _depth_multiplier;
    DepthwiseConvolutionMethod _method;
    QuantizationInfo           _out_quant_info;
    ActivationLayerInfo        _fused_activation;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_DEPTHWISE_CONVOLUTION_LAYER_NODE_H */
