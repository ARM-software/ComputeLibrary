/*
 * Copyright (c) 2018-2020 ARM Limited.
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
     * @param[in] op             Element-wise operation to perform
     * @param[in] out_quant_info (Optional) Output quantization information
     * @param[in] c_policy       (Optional) Convert policy used for the operation
     * @param[in] r_policy       (Optional) Rounding policy used for the operation
     */
    EltwiseLayerNode(EltwiseOperation op, QuantizationInfo out_quant_info = QuantizationInfo(), ConvertPolicy c_policy = ConvertPolicy::SATURATE, RoundingPolicy r_policy = RoundingPolicy::TO_ZERO);
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

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    EltwiseOperation _op;
    QuantizationInfo _out_quant_info;
    ConvertPolicy    _convert_policy;
    RoundingPolicy   _rounding_policy;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ELTWISE_LAYER_NODE_H */
