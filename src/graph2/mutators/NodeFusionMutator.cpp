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
#include "arm_compute/graph2/mutators/NodeFusionMutator.h"

#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/Logger.h"
#include "arm_compute/graph2/nodes/Nodes.h"

#include "arm_compute/core/utils/misc/Cast.h"

namespace arm_compute
{
namespace graph2
{
namespace detail
{
void fuse_batch_norm_with_activation(Graph &g)
{
    // Not interested in the order of nodes
    for(auto &node : g.nodes())
    {
        // Check if the node is batch norm and not a branching node
        if(node && node->type() == NodeType::BatchNormalizationLayer && node->output_edges().size() == 1)
        {
            auto output_edge_id = *node->output_edges().begin();
            auto output_edge    = g.edge(output_edge_id);
            // Check if following node is an activation layer node
            if((output_edge != nullptr) && (output_edge->consumer() != nullptr) && (output_edge->consumer()->type() == NodeType::ActivationLayer))
            {
                ARM_COMPUTE_LOG_GRAPH_VERBOSE("Fusing Batch Normalization node with ID : " << output_edge->producer_id()
                                              << " with Activation Layer node with ID : " << output_edge->consumer_id() << std::endl);

                auto *bn_node  = arm_compute::utils::cast::polymorphic_downcast<BatchNormalizationLayerNode *>(output_edge->producer());
                auto *act_node = arm_compute::utils::cast::polymorphic_downcast<ActivationLayerNode *>(output_edge->consumer());

                // Get driving nodes of activation node
                std::vector<NodeIdxPair> act_driving_nodes;
                for(auto &act_output_edge_id : act_node->output_edges())
                {
                    auto act_output_edge = g.edge(act_output_edge_id);
                    if(act_output_edge != nullptr)
                    {
                        ARM_COMPUTE_ERROR_ON(act_output_edge->consumer() == nullptr);
                        act_driving_nodes.push_back({ act_output_edge->consumer_id(), act_output_edge->consumer_idx() });
                    }
                }

                // Set activation info to batch normalization
                bn_node->set_fused_activation(act_node->activation_info());

                // Remove activation node
                g.remove_node(act_node->id());

                // Update batch normalization node outputs
                for(auto &driving_node : act_driving_nodes)
                {
                    g.add_connection(bn_node->id(), 0, driving_node.node_id, driving_node.index);
                }
            }
        }
    }
}
} // namespace detail

const char *NodeFusionMutator::name()
{
    return "NodeFusionMutator";
}

void NodeFusionMutator::mutate(Graph &g)
{
    detail::fuse_batch_norm_with_activation(g);
}
} // namespace graph2
} // namespace arm_compute
