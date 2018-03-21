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
#ifndef __ARM_COMPUTE_GRAPH2_BATCH_NORMALIZATION_LAYER_NODE_H__
#define __ARM_COMPUTE_GRAPH2_BATCH_NORMALIZATION_LAYER_NODE_H__

#include "arm_compute/graph2/INode.h"

namespace arm_compute
{
namespace graph2
{
/** Batch Normalization Layer node */
class BatchNormalizationLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] epsilon          (Optional) Epsilon parameter. Defaults to 1.f
     * @param[in] fused_activation (Optional) Fused activation layer. Disabled if not specified
     */
    BatchNormalizationLayerNode(float epsilon = 1.f, ActivationLayerInfo fused_activation = ActivationLayerInfo());
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

    // Inherited overridden methods:
    Status           validate() override;
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    float               _epsilon;
    ActivationLayerInfo _fused_activation;
};
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_BATCH_NORMALIZATION_LAYER_NODE_H__ */
