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
#ifndef ARM_COMPUTE_GRAPH_SLICE_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_SLICE_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

#include <tuple>

namespace arm_compute
{
namespace graph
{
/** Slice Layer node */
class SliceLayerNode final : public INode
{
public:
    /** Default Constructor
     *
     * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     */
    SliceLayerNode(const Coordinates &starts, const Coordinates &ends);
    /** Computes slice layer output descriptor
     *
     * @param[in] input_descriptor Descriptor of the input tensor
     * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     *
     * @return  Output descriptor
     */
    static TensorDescriptor compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                      const Coordinates &starts, const Coordinates &ends);
    /** Start coordinates accessor
     *
     * @return Start coordinates of the dimensions
     */
    Coordinates starts() const;
    /** End coordinates accessor
     *
     * @return End coordinates of the dimensions
     */
    Coordinates ends() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    Coordinates _starts;
    Coordinates _ends;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_SLICE_LAYER_NODE_H */
