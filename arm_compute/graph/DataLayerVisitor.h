/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_DATALAYERPRINTER_H
#define ARM_COMPUTE_GRAPH_DATALAYERPRINTER_H

#include "arm_compute/graph/IGraphPrinter.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
/** Graph printer visitor. */
class DataLayerVisitor final : public DefaultNodeVisitor
{
public:
    using LayerData = std::map<std::string, std::string>;
    /** Default Constructor **/
    DataLayerVisitor() = default;

    const LayerData &layer_data() const;

    // Reveal Parent method
    using DefaultNodeVisitor::visit;
    // Inherited methods overridden
    void visit(ConvolutionLayerNode &n) override;
    void visit(DepthwiseConvolutionLayerNode &n) override;
    void visit(FusedConvolutionBatchNormalizationNode &n) override;
    void visit(FusedConvolutionBatchNormalizationWithPostOpsNode &n) override;
    void visit(FusedDepthwiseConvolutionBatchNormalizationNode &n) override;
    void visit(OutputNode &n) override;

    void default_visit(INode &n) override;

private:
    LayerData _layer_data{};
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_DATALAYERPRINTER_H */
