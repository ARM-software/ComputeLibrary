/*
 * Copyright (c) 2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_STACK_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_STACK_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Stack Layer node */
class StackLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] total_nodes Number of nodes that will get stacked
     * @param[in] axis        Axis alogn which to stack the input tensors
     */
    StackLayerNode(unsigned int total_nodes, int axis);
    /** Computes stack output descriptor
     *
     * @param[in] input_descriptors Input descriptors
     * @param[in] axis              Axis along which to stack the input tensors
     *
     * @return Expected output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const std::vector<TensorDescriptor> &input_descriptors, int axis);
    /** Stack axis parameter accessor
     *
     * @return Stack axis
     */
    int axis() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    unsigned int _total_nodes;
    int          _axis;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_STACK_LAYER_NODE_H */
