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
#ifndef ARM_COMPUTE_GRAPH_PAD_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_PAD_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Pad Layer node */
class PadLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] padding   The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                      specifies the front and the end padding in the i-th dimension.
     * @param[in] pad_value Padding value to be used. Defaults to 0
     */
    PadLayerNode(const PaddingList &padding, PixelValue pad_value = PixelValue());
    /** Padding list accessor
     *
     * @return Padding list
     */
    const PaddingList &padding() const;
    /** Padding value accessor
     *
     * @return Padding value
     */
    PixelValue pad_value() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

public:
    static constexpr NodeType node_type = NodeType::PadLayer;

private:
    PaddingList _padding;
    PixelValue  _pad_value;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_PAD_LAYER_NODE_H */
