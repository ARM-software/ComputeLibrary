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
#include "arm_compute/graph/printers/DotGraphPrinter.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/nodes/Nodes.h"

namespace arm_compute
{
namespace graph
{
void DotGraphVisitor::visit(ActivationLayerNode &n)
{
    std::stringstream ss;
    ss << n.activation_info().activation();
    _info = ss.str();
}

void DotGraphVisitor::visit(BatchNormalizationLayerNode &n)
{
    std::stringstream ss;
    ss << (n.fused_activation().enabled() ? to_string(n.fused_activation().activation()) : "");
    _info = ss.str();
}

void DotGraphVisitor::visit(ConcatenateLayerNode &n)
{
    std::stringstream ss;
    ss << "Enabled: " << n.is_enabled();
    ss << R"( \n )";
    ss << "Axis: " << n.concatenation_axis();
    _info = ss.str();
}

void DotGraphVisitor::visit(ConvolutionLayerNode &n)
{
    std::stringstream ss;
    ss << n.convolution_method();
    _info = ss.str();
}

void DotGraphVisitor::visit(DepthwiseConvolutionLayerNode &n)
{
    std::stringstream ss;
    ss << n.depthwise_convolution_method();
    _info = ss.str();
}

void DotGraphVisitor::visit(EltwiseLayerNode &n)
{
    std::stringstream ss;
    ss << n.eltwise_operation();
    _info = ss.str();
}

void DotGraphVisitor::visit(FusedConvolutionBatchNormalizationNode &n)
{
    ARM_COMPUTE_UNUSED(n);
    std::stringstream ss;
    ss << "FusedConvolutionBatchNormalizationNode";
    _info = ss.str();
}

void DotGraphVisitor::visit(NormalizationLayerNode &n)
{
    std::stringstream ss;
    ss << n.normalization_info().type();
    _info = ss.str();
}

void DotGraphVisitor::visit(PoolingLayerNode &n)
{
    std::stringstream ss;
    ss << n.pooling_info().pool_type();
    ss << R"( \n )";
    ss << n.pooling_info().pool_size();
    ss << R"( \n )";
    ss << n.pooling_info().pad_stride_info();
    _info = ss.str();
}

void DotGraphVisitor::default_visit()
{
    _info.clear();
}

const std::string &DotGraphVisitor::info() const
{
    return _info;
}

void DotGraphPrinter::print(const Graph &g, std::ostream &os)
{
    // Print header
    print_header(g, os);

    // Print nodes
    print_nodes(g, os);

    // Print edges
    print_edges(g, os);

    // Print footer
    print_footer(g, os);
}

void DotGraphPrinter::print_header(const Graph &g, std::ostream &os)
{
    // Print graph name
    std::string graph_name = (g.name().empty()) ? "Graph" : g.name();
    os << "digraph " << graph_name << "{\n";
}

void DotGraphPrinter::print_footer(const Graph &g, std::ostream &os)
{
    ARM_COMPUTE_UNUSED(g);
    os << "}\n";
}

void DotGraphPrinter::print_nodes(const Graph &g, std::ostream &os)
{
    for(const auto &n : g.nodes())
    {
        if(n)
        {
            // Output node id
            std::string node_id = std::string("n") + support::cpp11::to_string(n->id());
            os << node_id << " ";

            // Output label
            n->accept(_dot_node_visitor);

            std::string name             = n->name().empty() ? node_id : n->name();
            auto        node_description = _dot_node_visitor.info();

            os << R"([label = ")" << name << R"( \n )" << n->assigned_target() << R"( \n )" << node_description << R"("])";
            os << ";\n";
        }
    }
}

void DotGraphPrinter::print_edges(const Graph &g, std::ostream &os)
{
    for(const auto &e : g.edges())
    {
        if(e)
        {
            std::string source_node_id = std::string("n") + support::cpp11::to_string(e->producer_id());
            std::string sink_node_id   = std::string("n") + support::cpp11::to_string(e->consumer_id());
            os << source_node_id << " -> " << sink_node_id << " ";
            const Tensor *t = e->tensor();
            ARM_COMPUTE_ERROR_ON(t == nullptr);
            os << R"([label = ")" << t->desc().shape << R"( \n )" << t->desc().data_type << R"( \n )" << t->desc().layout << R"("])";
            os << ";\n";
        }
    }
}
} // namespace graph
} // namespace arm_compute
