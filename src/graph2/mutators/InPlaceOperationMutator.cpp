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
#include "arm_compute/graph2/mutators/InPlaceOperationMutator.h"

#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/Logger.h"

namespace arm_compute
{
namespace graph2
{
const char *InPlaceOperationMutator::name()
{
    return "InPlaceOperationMutator";
}

void InPlaceOperationMutator::mutate(Graph &g)
{
    std::set<NodeType> in_place_nodes = { NodeType::BatchNormalizationLayer, NodeType::ActivationLayer };

    // Not interested in the order of nodes
    for(auto &node : g.nodes())
    {
        if(node && in_place_nodes.find(node->type()) != std::end(in_place_nodes))
        {
            // Get input edge
            Edge *input_edge = node->input_edge(0);

            // Check if parent has a single output if yes then force in place calculation else not
            if((input_edge != nullptr) && (input_edge->producer() != nullptr) && (input_edge->producer()->output_edges().size() == 1))
            {
                ARM_COMPUTE_LOG_GRAPH_VERBOSE("Switching to in-place computation for the node with ID : "
                                              << node->id() << " and name : " << node->name() << std::endl);
                // Update output
                auto tensor = input_edge->tensor();
                node->set_output_tensor(tensor->id(), 0);
            }
        }
    }
}
} // namespace graph2
} // namespace arm_compute
