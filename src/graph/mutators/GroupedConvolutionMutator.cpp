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
#include "arm_compute/graph/mutators/GroupedConvolutionMutator.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphBuilder.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/utils/misc/Cast.h"

#include <set>

namespace arm_compute
{
namespace graph
{
namespace
{
NodeID create_grouped_convolution(Graph &g, const NodeParams &params, NodeIdxPair input, NodeID weights, NodeID bias,
                                  PadStrideInfo conv_info, ConvolutionMethod method, FastMathHint fast_math_hint, unsigned int num_groups)
{
    bool has_bias = (bias != EmptyNodeID);

    // Split input
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
    const unsigned int     input_idx         = get_dimension_idx(input_tensor_desc, DataLayoutDimension::CHANNEL);
    NodeID                 input_split       = GraphBuilder::add_split_node(g, params, input, num_groups, input_idx);

    // Split weights
    const TensorDescriptor weights_tensor_desc = get_tensor_descriptor(g, g.node(weights)->outputs()[0]);
    const unsigned int     batch_idx           = get_dimension_idx(weights_tensor_desc, DataLayoutDimension::BATCHES);
    NodeID                 weights_split       = GraphBuilder::add_split_node(g, params, { weights, 0 }, num_groups, batch_idx);

    // Split bias
    NodeID bias_split = EmptyNodeID;
    if(has_bias)
    {
        // Split bias
        bias_split = GraphBuilder::add_split_node(g, params, { bias, 0 }, num_groups, 0);
    }

    std::vector<NodeIdxPair> convolution_outputs;
    for(unsigned int i = 0; i < num_groups; ++i)
    {
        NodeParams group_params = params;
        NodeID     conv_nid     = g.add_node<ConvolutionLayerNode>(conv_info, 1, method, fast_math_hint);
        g.add_connection(input_split, i, conv_nid, 0);
        g.add_connection(weights_split, i, conv_nid, 1);
        if(has_bias)
        {
            g.add_connection(bias_split, i, conv_nid, 2);
        }

        // Add group name
        if(!group_params.name.empty())
        {
            group_params.name.append("_g" + arm_compute::support::cpp11::to_string(i));
        }

        // Set node parameters
        INode *node = g.node(conv_nid);
        ARM_COMPUTE_ERROR_ON(node == nullptr);
        node->set_common_node_parameters(group_params);

        convolution_outputs.push_back({ conv_nid, 0 });
    }

    // Depth concatenate output
    return GraphBuilder::add_concatenate_node(g, params, convolution_outputs, DataLayoutDimension::CHANNEL);
}
} // namespace

const char *GroupedConvolutionMutator::name()
{
    return "GroupedConvolutionMutator";
}

void GroupedConvolutionMutator::mutate(Graph &g)
{
    // Early exit if no Convolution layers exist in graph
    if(g.nodes(NodeType::ConvolutionLayer).empty())
    {
        return;
    }

    // Total nodes
    size_t total_nodes = g.nodes().size();

    // Iterate over convolution nodes
    for(unsigned int i = 0; i < total_nodes; ++i)
    {
        INode *node = g.node(i);
        if(node != nullptr && node->type() == NodeType::ConvolutionLayer && arm_compute::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(node)->num_groups() != 1)
        {
            // Validate node
            backends::IDeviceBackend *backend = backends::BackendRegistry::get().find_backend(node->assigned_target());
            Status                    status  = backend->validate_node(*node);

            // If grouped convolution is not supported
            if(!bool(status))
            {
                // Down-cast node
                auto *conv_node = arm_compute::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(node);

                // Get internal convolution info
                // TODO (geopin01) : Create a descriptor
                const PadStrideInfo     conv_info       = conv_node->convolution_info();
                const ConvolutionMethod conv_method     = conv_node->convolution_method();
                const FastMathHint      fast_math_hint  = conv_node->fast_math_hint();
                const unsigned int      num_groups      = conv_node->num_groups();
                const NodeParams        params          = conv_node->common_node_params();
                const Target            assigned_target = conv_node->assigned_target();

                // Extract node ids
                const NodeID input_id   = conv_node->input_id(0);
                const NodeID weights_id = conv_node->input_id(1);
                const NodeID bias_id    = conv_node->input_id(2);

                // Get driving nodes
                std::vector<NodeIdxPair> driving_nodes = get_driving_nodes(*node);

                // Extract activation node accessor if any
                auto node_accessor = conv_node->output(0)->extract_accessor();

                // Current max tensor and node id
                TensorID latest_tid = g.tensors().size();
                NodeID   latest_nid = g.nodes().size();

                // Create grouped convolution node
                NodeID grouped_conv_id = create_grouped_convolution(g, params, { input_id, 0 }, weights_id, bias_id,
                                                                    conv_info, conv_method, fast_math_hint, num_groups);

                // Remove convolution node
                g.remove_node(node->id());

                // Update batch normalization node outputs
                for(auto &driving_node : driving_nodes)
                {
                    g.add_connection(grouped_conv_id, 0, driving_node.node_id, driving_node.index);
                }

                // Update accessor to batch normalization node
                g.node(grouped_conv_id)->output(0)->set_accessor(std::move(node_accessor));

                // Configure new tensors and nodes
                std::for_each(g.tensors().begin() + latest_tid, g.tensors().end(), [](std::unique_ptr<Tensor> &t)
                {
                    configure_tensor(t.get());
                });
                std::for_each(g.nodes().begin() + latest_nid, g.nodes().end(), [&assigned_target](std::unique_ptr<INode> &n)
                {
                    if(n != nullptr)
                    {
                        n->set_assigned_target(assigned_target);
                    }
                });
            }
        }
    }
}
} // namespace graph
} // namespace arm_compute
