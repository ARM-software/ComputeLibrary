/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_FULLY_CONNECTED_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_FULLY_CONNECTED_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Fully Connected Layer node */
class FullyConnectedLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] num_outputs    Number of neurons in the layer
     * @param[in] out_quant_info (Optional) Output quantization info
     * @param[in] fc_info        (Optional) Additional information about the fully connected layer
     * @param[in] fast_math_hint (Optional) Fast math hint
     */
    FullyConnectedLayerNode(unsigned int            num_outputs,
                            QuantizationInfo        out_quant_info = QuantizationInfo(),
                            FullyConnectedLayerInfo fc_info        = FullyConnectedLayerInfo(),
                            FastMathHint            fast_math_hint = FastMathHint::Disabled);
    /** Sets the fast math fast hint
     *
     * @param[in] hint Hint to use for fullyconnected layer
     */
    void set_fast_math_hint(FastMathHint hint);
    /** Fast math hint accessor
     *
     * @return Fast math hint to be used by the node
     */
    FastMathHint fast_math_hint() const;
    /** Sets fused activation
     *
     * @param[in] fused_activation Fused activation to set
     */
    void set_fused_activation(ActivationLayerInfo fused_activation);
    /** Computes weights descriptor
     *
     * @warning Works for inputs with 1D batch space
     *
     * @param[in] input_descriptor   Input descriptor
     * @param[in] num_outputs        Number of output neurons
     * @param[in] fc_info            (Optional) Additional information about the fully connected layer
     * @param[in] weights_quant_info (Optional) Weights quantization info
     *
     * @return Weights descriptor
     */
    static TensorDescriptor compute_weights_descriptor(const TensorDescriptor &input_descriptor,
                                                       unsigned int            num_outputs,
                                                       FullyConnectedLayerInfo fc_info            = FullyConnectedLayerInfo(),
                                                       const QuantizationInfo &weights_quant_info = QuantizationInfo());
    /** Computes fully connected layer output descriptor
     *
     * @warning Works for inputs with 1D batch space
     *
     * @param[in] input_descriptor Input descriptor
     * @param[in] num_outputs      Number of output neurons
     * @param[in] out_quant_info   (Optional) Weights quantization info
     *
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                      unsigned int            num_outputs,
                                                      const QuantizationInfo &out_quant_info = QuantizationInfo());
    /** Fully connected layer addition information
     *
     * @return Additional information about the fully connected layer
     */
    FullyConnectedLayerInfo info() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

    static constexpr NodeType node_type = NodeType::FullyConnectedLayer;

private:
    unsigned int            _num_outputs;
    QuantizationInfo        _out_quant_info;
    FullyConnectedLayerInfo _info;
    FastMathHint            _fast_math_hint;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_FULLY_CONNECTED_LAYER_NODE_H */
