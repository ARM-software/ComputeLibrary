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
#ifndef __ARM_COMPUTE_GRAPH2_SPLIT_LAYER_NODE_H__
#define __ARM_COMPUTE_GRAPH2_SPLIT_LAYER_NODE_H__

#include "arm_compute/graph2/INode.h"

#include <tuple>

namespace arm_compute
{
namespace graph2
{
/** Split Layer node */
class SplitLayerNode final : public INode
{
public:
    /** Default Constructor
     *
     * @param[in] num_splits Number of splits
     * @param[in] axis       (Optional) Axis to split on. Supported axis >= 2. Defaults to 0
     */
    SplitLayerNode(unsigned int num_splits, unsigned int axis = 0);
    /** Computes split layer output shape
     *
     * @param[in] input_shape Shape of the input
     * @param[in] num_splits  Number of splits
     * @param[in] axis        Axis to perform the split on
     * @param[in] idx         Index of the split
     *
     * @return  A pair with the shape of the split and the starting coordinates
     */
    static std::pair<TensorShape, Coordinates> compute_output_shape(TensorShape input_shape, unsigned int num_splits, unsigned int axis, unsigned int idx);
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
    Status           validate() override;
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    unsigned int _num_splits;
    unsigned int _axis;
};
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_SPLIT_LAYER_NODE_H__ */
