/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_ARGMINMAX_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_ARGMINMAX_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Arg Min/Max Layer node */
class ArgMinMaxLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] op             Operation to perform: min or max
     * @param[in] axis           Axis along which to reduce. Supported reduction axis : 0,1,2,3
     * @param[in] out_data_type  (Optional) Output data type
     * @param[in] out_quant_info (Optional) Output quantization info
     */
    ArgMinMaxLayerNode(ReductionOperation op,
                       unsigned int       axis,
                       DataType           out_data_type  = DataType::UNKNOWN,
                       QuantizationInfo   out_quant_info = QuantizationInfo());
    /** Operator accessor
     *
     * @return The operator the layer performs: min or max
     */
    ReductionOperation reduction_operation() const;
    /** Axis accessor
     *
     * @return The axis along which the reduction is operating
     */
    unsigned int axis() const;
    /** Output data type accessor
     *
     * @return The output data type
     */
    DataType out_data_type() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

public:
    static constexpr NodeType node_type = NodeType::ArgMinMaxLayer;

private:
    ReductionOperation _op;
    unsigned int       _axis;
    DataType           _out_data_type;
    QuantizationInfo   _out_quant_info;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ARGMINMAX_LAYER_NODE_H */
