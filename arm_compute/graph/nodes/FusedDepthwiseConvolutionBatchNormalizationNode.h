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
#ifndef __ARM_COMPUTE_GRAPH_FUSED_DEPTHWISE_CONVOLUTION_BATCH_NORMALIZATION_NODE_H__
#define __ARM_COMPUTE_GRAPH_FUSED_DEPTHWISE_CONVOLUTION_BATCH_NORMALIZATION_NODE_H__

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Fused Depthwise Convolution Batch Normalization node */
class FusedDepthwiseConvolutionBatchNormalizationNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] epsilon          Epsilon parameter.
     * @param[in] info             Convolution layer attributes.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] method           (Optional) Convolution method to use
     * @param[in] fused_activation (Optional) Fused activation layer. Disabled if not specified
     */
    FusedDepthwiseConvolutionBatchNormalizationNode(float                      epsilon,
                                                    PadStrideInfo              info,
                                                    unsigned int               depth_multiplier,
                                                    DepthwiseConvolutionMethod method,
                                                    ActivationLayerInfo        fused_activation = ActivationLayerInfo());

    /** Sets the depthwise convolution layer method to use
     *
     * @param[in] method Method to use for depthwise convolution
     */
    void set_depthwise_convolution_method(DepthwiseConvolutionMethod method);

    /** Depthwise convolution layer method accessor
     *
     * @note This is an indication on which depthwise convolution layer implementation to use,
     *       if it fails to be created the library's heuristic approach will be used
     *
     * @return Depthwise convolution layer method to be used by the node
     */
    DepthwiseConvolutionMethod depthwise_convolution_method() const;

    /** Epsilon parameter accessor
     *
     * @return Epsilon parameter
     */
    float epsilon() const;

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

    /** Computes convolution output descriptor
     *
     * @param[in] input_descriptor   Input descriptor
     * @param[in] weights_descriptor Weights descriptor
     * @param[in] info               Convolution operation attributes
     * @param[in] depth_multiplier   Depth multiplier
     *
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                      const TensorDescriptor &weights_descriptor,
                                                      const PadStrideInfo    &info,
                                                      int                     depth_multiplier);

    /** Sets the convolution layer method to use
     *
     * @param[in] method Method to use for convolution
     */
    void set_convolution_method(ConvolutionMethod method);

    /** Depth multiplier accessor
     *
     * @return Depth multiplier
     */
    unsigned int depth_multiplier() const;

    /** Convolution metadata accessor
     *
     * @return Convolution information
     */
    PadStrideInfo convolution_info() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

public:
    static constexpr NodeType node_type = NodeType::FusedDepthwiseConvolutionBatchNormalizationLayer;

private:
    float _epsilon;

    PadStrideInfo              _info;
    unsigned int               _depth_multiplier;
    DepthwiseConvolutionMethod _method;
    ActivationLayerInfo        _fused_activation;
};

} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_FUSED_DEPTHWISE_CONVOLUTION_BATCH_NORMALIZATION_NODE_H__ */
