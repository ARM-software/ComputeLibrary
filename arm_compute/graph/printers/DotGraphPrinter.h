/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_DOTGRAPHPRINTER_H__
#define __ARM_COMPUTE_GRAPH_DOTGRAPHPRINTER_H__

#include "arm_compute/graph/IGraphPrinter.h"

#include "arm_compute/graph/INodeVisitor.h"

#include <string>

namespace arm_compute
{
namespace graph
{
/** Graph printer visitor. */
class DotGraphVisitor final : public DefaultNodeVisitor
{
public:
    /** Default Constructor **/
    DotGraphVisitor() = default;
    /** Returns the output information of the last visited node
     *
     * @return Information of the last visited node
     */
    const std::string &info() const;

    // Reveal parent method
    using DefaultNodeVisitor::visit;

    // Inherited methods overridden
    void visit(ActivationLayerNode &n) override;
    void visit(BatchNormalizationLayerNode &n) override;
    void visit(ConcatenateLayerNode &n) override;
    void visit(ConvolutionLayerNode &n) override;
    void visit(DepthwiseConvolutionLayerNode &n) override;
    void visit(EltwiseLayerNode &n) override;
    void visit(FusedConvolutionBatchNormalizationNode &n) override;
    void visit(NormalizationLayerNode &n) override;
    void visit(PoolingLayerNode &n) override;
    void default_visit() override;

private:
    std::string _info{};
};

/** Graph printer interface */
class DotGraphPrinter final : public IGraphPrinter
{
public:
    // Inherited methods overridden
    void print(const Graph &g, std::ostream &os) override;

private:
    /** Print dot graph header
     *
     * @param[in]  g  Graph
     * @param[out] os Output stream to use
     */
    void print_header(const Graph &g, std::ostream &os);
    /** Print dot graph footer
     *
     * @param[in]  g  Graph
     * @param[out] os Output stream to use
     */
    void print_footer(const Graph &g, std::ostream &os);
    /** Prints nodes in dot format
     *
     * @param[in]  g  Graph
     * @param[out] os Output stream to use
     */
    void print_nodes(const Graph &g, std::ostream &os);
    /** Prints edges in dot format
     *
     * @param[in]  g  Graph
     * @param[out] os Output stream to use
     */
    void print_edges(const Graph &g, std::ostream &os);

private:
    DotGraphVisitor _dot_node_visitor = {};
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_DOTGRAPHPRINTER_H__ */
