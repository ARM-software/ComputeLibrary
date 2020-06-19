/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/graph/mutators/InPlaceOperationMutator.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Logger.h"

namespace arm_compute
{
namespace graph
{
namespace
{
// Check if the output edges of the parent node are separate tensors. If not,
// it means the same output is connected to multiple nodes and computations on
// these nodes cannot be done in-place.
bool output_edges_are_separate_tensors(Graph &g, const Edge *input_edge)
{
    const auto parent_node   = input_edge->producer();
    const auto input_tensor  = input_edge->tensor();
    const auto input_edge_id = input_edge->id();

    if(parent_node == nullptr)
    {
        return false;
    }

    const auto output_edges = parent_node->output_edges();

    // If the output is connected to only one edge, then computations can
    // be done in-place.
    if(output_edges.size() == 1)
    {
        return true;
    }

    return std::all_of(output_edges.begin(),
                       output_edges.end(),
                       [&](const EdgeID & edge_id)
    {
        // Skip check on current input edge
        if(edge_id == input_edge_id)
        {
            return true;
        }

        auto edge = g.edge(edge_id);
        return edge->tensor() != input_tensor;
    });
}
} // namespace

const char *InPlaceOperationMutator::name()
{
    return "InPlaceOperationMutator";
}

IGraphMutator::MutationType InPlaceOperationMutator::type() const
{
    return IGraphMutator::MutationType::Backend;
}

void InPlaceOperationMutator::mutate(Graph &g)
{
    std::set<NodeType> in_place_nodes =
    {
        NodeType::ActivationLayer,
        NodeType::BatchNormalizationLayer,
        NodeType::EltwiseLayer,
        NodeType::UnaryEltwiseLayer,
        NodeType::PrintLayer
    };

    // Not interested in the order of nodes
    for(auto &node : g.nodes())
    {
        if(node && in_place_nodes.find(node->type()) != std::end(in_place_nodes))
        {
            // Get input edge
            Edge *input_edge = node->input_edge(0);

            // Check if parent has a single output if yes then force in place calculation else not
            if((input_edge != nullptr) && output_edges_are_separate_tensors(g, input_edge))
            {
                // Get current and new output tensors
                auto current_output_tensor = node->output(0);
                auto new_output_tensor     = input_edge->tensor();

                ARM_COMPUTE_ERROR_ON(current_output_tensor == nullptr || new_output_tensor == nullptr);

                // Prevent in-place operation if there is an accessor bound to the in-place tensor or quantization info are different
                if(new_output_tensor->accessor() != nullptr || current_output_tensor->desc().quant_info != new_output_tensor->desc().quant_info)
                {
                    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented in-place operation as there is an accessor bound to the input tensor or the quantization info are different.\n");
                }
                else
                {
                    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Switching to in-place computation for the node with ID : "
                                                  << node->id() << " and name : " << node->name() << std::endl);
                    // Update accessor
                    new_output_tensor->set_accessor(current_output_tensor->extract_accessor());
                    // Update output
                    node->set_output_tensor(new_output_tensor->id(), 0);
                }
            }
        }
    }
}
} // namespace graph
} // namespace arm_compute
