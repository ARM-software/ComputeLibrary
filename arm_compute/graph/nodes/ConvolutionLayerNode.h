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
#ifndef __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_NODE_H__
#define __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_NODE_H__

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Convolution Layer node */
class ConvolutionLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] info           Convolution layer attributes
     * @param[in] method         (Optional) Convolution method to use
     * @param[in] fast_math_hint (Optional) Fast math hint
     * @param[in] out_quant_info (Optional) Output quantization info
     */
    ConvolutionLayerNode(PadStrideInfo     info,
                         ConvolutionMethod method         = ConvolutionMethod::Default,
                         FastMathHint      fast_math_hint = FastMathHint::Disabled,
                         QuantizationInfo  out_quant_info = QuantizationInfo());
    /** Sets the convolution layer method to use
     *
     * @param[in] method Method to use for convolution
     */
    void set_convolution_method(ConvolutionMethod method);
    /** Convolution layer method accessor
     *
     * @note This is an indication on which convolution layer implementation to use,
     *       if it fails to be created the library's heuristic approach will be used
     *
     * @return Convolution layer method to be used by the node
     */
    ConvolutionMethod convolution_method() const;
    /** Sets the fast math fast hint
     *
     * @param[in] hint Hint to use for convolution
     */
    void set_fast_math_hint(FastMathHint hint);
    /** Fast math hint accessor
     *
     * @return Fast math hint to be used by the node
     */
    FastMathHint fast_math_hint() const;
    /** Convolution metadata accessor
     *
     * @return Convolution information
     */
    PadStrideInfo convolution_info() const;
    /** Computes convolution output descriptor
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
    PadStrideInfo     _info;
    ConvolutionMethod _method;
    FastMathHint      _fast_math_hint;
    QuantizationInfo  _out_quant_info;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_CONVOLUTION_LAYER_NODE_H__ */
