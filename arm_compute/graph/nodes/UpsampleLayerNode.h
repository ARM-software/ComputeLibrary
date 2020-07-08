/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_UPSAMPLE_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_UPSAMPLE_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Upsample Layer node */
class UpsampleLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] info              Stride info
     * @param[in] upsampling_policy Upsampling policy
     */
    UpsampleLayerNode(Size2D info, InterpolationPolicy upsampling_policy);
    /** Stride info metadata accessor
     *
     * @return The stride info of the layer
     */
    Size2D info() const;
    /** Upsampling policy metadata accessor
     *
     * @return The upsampling policy of the layer
     */
    InterpolationPolicy upsampling_policy() const;
    /** Computes upsample output descriptor
     *
     * @param[in] input_descriptor Input descriptor
     * @param[in] info             Stride information
     *
     * @return Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor, Size2D info);

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    Size2D              _info;
    InterpolationPolicy _upsampling_policy;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_UPSAMPLE_LAYER_NODE_H */
