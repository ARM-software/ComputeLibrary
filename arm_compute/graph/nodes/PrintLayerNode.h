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
#ifndef ARM_COMPUTE_GRAPH_PRINT_LAYER_NODE_H
#define ARM_COMPUTE_GRAPH_PRINT_LAYER_NODE_H

#include "arm_compute/graph/INode.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

namespace graph
{
/** Print Layer node */
class PrintLayerNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] stream      Output stream.
     * @param[in] format_info (Optional) Format info.
     * @param[in] transform   (Optional) Input transform function.
     */
    PrintLayerNode(std::ostream &stream, const IOFormatInfo &format_info = IOFormatInfo(), const std::function<ITensor *(ITensor *)> transform = nullptr);

    /** Stream metadata accessor
     *
     * @return Print Layer stream
     */
    std::ostream &stream() const;

    /** Formatting metadata accessor
     *
     * @return Print Layer format info
     */
    const IOFormatInfo format_info() const;

    /** Transform function metadata accessor
     *
     * @return Print Layer transform function
     */
    const std::function<ITensor *(ITensor *)> transform() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void accept(INodeVisitor &v) override;

private:
    std::ostream                             &_stream;
    const IOFormatInfo                        _format_info;
    const std::function<ITensor *(ITensor *)> _transform;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_PRINT_LAYER_NODE_H */
