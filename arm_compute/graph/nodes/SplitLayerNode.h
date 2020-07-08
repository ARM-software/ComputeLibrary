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
#ifndef ARM_COMPUTE_GRAPH_SPLIT_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_SPLIT_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

#include <tuple>

namespace arm_compute
{
namespace graph
{
/** Split Layer node */
class SplitLayerNode final : public INode
{
public:
    /** Default Constructor
     *
     * @param[in] num_splits  Number of splits
     * @param[in] axis        (Optional) Axis to split on. Defaults to 0
     * @param[in] size_splits (Optional) The sizes of each output tensor along the split dimension.
     *                        Must sum to the dimension of value along split_dim.
     *                        Can contain one -1 indicating that dimension is to be inferred.
     */
    SplitLayerNode(unsigned int num_splits, int axis = 0, std::vector<int> size_splits = std::vector<int>());
    /** Computes split layer output descriptor
     *
     * @param[in] input_descriptor Descriptor of the input tensor
     * @param[in] num_splits       Number of splits
     * @param[in] axis             Axis to perform the split on
     * @param[in] idx              Index of the split
     *
     * @return  A pair with the descriptor of the split and the starting coordinates
     */
    std::pair<TensorDescriptor, Coordinates> compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                       unsigned int num_splits, int axis, unsigned int idx);
    /** Number of splits accessor
     *
     * @return Number of splits
     */
    unsigned int num_splits() const;
    /** Split axis accessor
     *
     * @return Split axis
     */
    unsigned int axis() const;

    // Inherited overridden methods:
    Status           validate() const override;
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    unsigned int     _num_splits;
    int              _axis;
    std::vector<int> _size_splits;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_SPLIT_LAYER_NODE_H */
