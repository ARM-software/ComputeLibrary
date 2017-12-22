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
#ifndef __ARM_COMPUTE_GRAPH2_DEPTH_CONCATENATE_LAYER_NODE_H__
#define __ARM_COMPUTE_GRAPH2_DEPTH_CONCATENATE_LAYER_NODE_H__

#include "arm_compute/graph2/INode.h"

namespace arm_compute
{
namespace graph2
{
class DepthConcatenateLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] total_nodes Number of nodes that will get concatenated
     */
    DepthConcatenateLayerNode(unsigned int total_nodes);
    /** Computes depth concatenations output shape
     *
     * @param input_shapes   Shapes of the inputs
     *
     * @return Expected output shape
     */
    static TensorShape compute_output_shape(const std::vector<TensorShape> &input_shapes);
    /** Disables or not the depth concatenate node
     *
     * @warning This is used when depth concatenate is performed with sub-tensors,
     *          where this node is used as a placeholder.
     *
     * @param[in] is_enabled If true a backend function is created to perform the depth concatenation (involves copying),
     *                       while if false, no function is created and we assume that subtensors are properly set to simulate
     *                       a no copy operation.
     */
    void set_enabled(bool is_enabled);
    /** Enabled parameter accessor
     *
     * @return True if a backend function is to be created else false
     */
    bool is_enabled() const;

    // Inherited overridden methods:
    Status           validate() override;
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    unsigned int _total_nodes;
    bool         _is_enabled;
};
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_DEPTH_CONCATENATE_LAYER_NODE_H__ */
