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
#include "arm_compute/graph/mutators/NodeFusionMutator.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/utils/misc/Cast.h"

#include <set>

namespace arm_compute
{
namespace graph
{
namespace detail
{
template <typename N>
void fuse_node_with_activation(Graph                              &g,
                               const std::set<Activation>         &supported_fused_activations,
                               std::function<bool(INode &)> const &prec)
{
    // Not interested in the order of nodes
    for(auto &node : g.nodes())
    {
        // Check if the node is of type N and not a branching node
        if(node && node->type() == N::node_type && node->output_edges().size() == 1)
        {
            auto output_edge_id = *node->output_edges().begin();
            auto output_edge    = g.edge(output_edge_id);
            // Check if following node is an activation layer node
            if((output_edge != nullptr) && (output_edge->consumer() != nullptr) && (output_edge->consumer()->type() == NodeType::ActivationLayer))
            {
                auto *n_node   = arm_compute::utils::cast::polymorphic_downcast<N *>(output_edge->producer());
                auto *act_node = arm_compute::utils::cast::polymorphic_downcast<ActivationLayerNode *>(output_edge->consumer());

                ARM_COMPUTE_ERROR_ON(act_node->output(0) == nullptr || n_node->output(0) == nullptr);

                // Check given precondition
                if(!prec(*n_node))
                {
                    continue;
                }
                // Check if activation is supported for fusion
                if(supported_fused_activations.count(act_node->activation_info().activation()) == 0)
                {
                    continue;
                }

                ARM_COMPUTE_LOG_GRAPH_VERBOSE("Fusing node with ID : " << output_edge->producer_id()
                                              << " with Activation Layer node with ID : " << output_edge->consumer_id() << std::endl);

                // Prevent fusion if fused node has an output accessor
                if(n_node->output(0)->accessor() == nullptr)
                {
                    // Get driving nodes of activation node
                    std::vector<NodeIdxPair> act_driving_nodes = get_driving_nodes(*act_node);

                    // Set activation info to fused node
                    n_node->set_fused_activation(act_node->activation_info());

                    // Extract activation node accessor if any
                    auto act_node_accessor = act_node->output(0)->extract_accessor();

                    // Remove activation node
                    g.remove_node(act_node->id());

                    // Update fused node outputs
                    for(auto &driving_node : act_driving_nodes)
                    {
                        g.add_connection(n_node->id(), 0, driving_node.node_id, driving_node.index);
                    }

                    // Update accessor to fused node
                    n_node->output(0)->set_accessor(std::move(act_node_accessor));
                }
                else
                {
                    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of node with activation due to the presence of an output accessor\n");
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
    // Supported activations when fusing
    const std::set<Activation> supported_fused_activations = { Activation::RELU, Activation::BOUNDED_RELU, Activation::LU_BOUNDED_RELU };

    // Preconditions
    auto empty_prec = [](INode & n)
    {
        return true;
    };
    auto qs8_prec = [](INode & n)
    {
        ARM_COMPUTE_ERROR_ON(n.output(0) == nullptr);
        return n.output(0)->desc().data_type == DataType::QASYMM8;
    };

    // Fusion mutations
    detail::fuse_node_with_activation<BatchNormalizationLayerNode>(g, supported_fused_activations, empty_prec);
    detail::fuse_node_with_activation<ConvolutionLayerNode>(g, supported_fused_activations, empty_prec);
    detail::fuse_node_with_activation<DepthwiseConvolutionLayerNode>(g, supported_fused_activations, qs8_prec);
}
} // namespace graph
} // namespace arm_compute
