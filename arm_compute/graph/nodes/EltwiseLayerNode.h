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
#ifndef ARM_COMPUTE_GRAPH_ELTWISE_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_ELTWISE_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Eltwise Layer node */
class EltwiseLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] descriptor Containing information for the node described in @ref descriptors::EltwiseLayerDescriptor
     */
    EltwiseLayerNode(const descriptors::EltwiseLayerDescriptor &descriptor);
    /** Eltwise operation accessor
     *
     * @return Eltwise operation that is to be performed by the node
     */
    EltwiseOperation eltwise_operation() const;

    /** Convert policy accessor
     *
     * @return Convert policy that is used in the node
     */
    ConvertPolicy convert_policy() const;

    /** Rounding policy accessor
     *
     * @return Convert policy that is used in the node
     */
    RoundingPolicy rounding_policy() const;

    /** Returns fused activation
     *
     * @return Fused activation
     */
    ActivationLayerInfo fused_activation() const;

    /** Returns output quantization info
     *
     * @return Output quantization info
     */
    QuantizationInfo output_quant_info() const;

    /** Sets fused activation
     *
     * @param[in] fused_activation Fused activation to set
     */
    void set_fused_activation(ActivationLayerInfo fused_activation);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

    static constexpr NodeType node_type = NodeType::EltwiseLayer;

private:
    descriptors::EltwiseLayerDescriptor descriptor;
};

/** Unary Eltwise Layer node */
class UnaryEltwiseLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] descriptor Containing information for the node described in @ref descriptors::EltwiseLayerDescriptor
     */
    UnaryEltwiseLayerNode(const descriptors::UnaryEltwiseLayerDescriptor &descriptor);
    /** Unary eltwise layer descriptor
     *
     * @return Unary eltwise layer descriptor which containing information
     */
    descriptors::UnaryEltwiseLayerDescriptor eltwise_descriptor() const;

    /** Sets fused activation
     *
     * @param[in] fused_activation Fused activation to set
     */
    void set_fused_activation(ActivationLayerInfo fused_activation);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

    static constexpr NodeType node_type = NodeType::UnaryEltwiseLayer;

private:
    descriptors::UnaryEltwiseLayerDescriptor descriptor;
};

} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ELTWISE_LAYER_NODE_H */
