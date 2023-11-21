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
#include "arm_compute/graph/mutators/NodeExecutionMethodMutator.h"

#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/nodes/Nodes.h"
#include "arm_compute/graph/Utils.h"

#include "support/Cast.h"

namespace arm_compute
{
namespace graph
{
namespace
{
/** Runs a default setter function on a given types of nodes
 *
 * @tparam Setter Setter function to run
 *
 * @param[in, out] g         Graph to extract the nodes from
 * @param[in]      node_type Node type
 * @param[in]      setter    Setter function
 */
template <typename Setter>
void set_default_on_invalid_method(Graph &g, NodeType node_type, Setter &&setter)
{
    const std::vector<NodeID> &node_ids = g.nodes(node_type);
    for (auto &node_id : node_ids)
    {
        INode *node = g.node(node_id);
        if (node != nullptr)
        {
            // Validate node
            backends::IDeviceBackend &backend = backends::BackendRegistry::get().get_backend(node->assigned_target());
            Status                    status  = backend.validate_node(*node);

            // Set default execution method in case of failure
            if (!bool(status))
            {
                setter(node);
            }
        }
    }
}
} // namespace

const char *NodeExecutionMethodMutator::name()
{
    return "NodeExecutionMethodMutator";
}

IGraphMutator::MutationType NodeExecutionMethodMutator::type() const
{
    return IGraphMutator::MutationType::Backend;
}

void NodeExecutionMethodMutator::mutate(Graph &g)
{
    // Convolution Layer
    set_default_on_invalid_method(g, NodeType::ConvolutionLayer,
                                  [](INode *n)
                                  {
                                      ARM_COMPUTE_LOG_GRAPH_INFO("Switched ConvolutionLayer method of node with ID : "
                                                                 << n->id() << " and Name: " << n->name() << std::endl);
                                      auto *casted_node =
                                          arm_compute::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(n);
                                      casted_node->set_convolution_method(ConvolutionMethod::Default);
                                  });

    // Depthwise Convolution Layer
    set_default_on_invalid_method(
        g, NodeType::DepthwiseConvolutionLayer,
        [](INode *n)
        {
            ARM_COMPUTE_LOG_GRAPH_INFO("Switched Depthwise ConvolutionLayer method of node with ID : "
                                       << n->id() << " and Name: " << n->name() << std::endl);
            auto *casted_node = arm_compute::utils::cast::polymorphic_downcast<DepthwiseConvolutionLayerNode *>(n);
            casted_node->set_depthwise_convolution_method(DepthwiseConvolutionMethod::Default);
        });
}
} // namespace graph
} // namespace arm_compute
