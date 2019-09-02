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
#include "arm_compute/graph/mutators/NodeFusionMutator.h"

#include "arm_compute/graph/GraphBuilder.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/nodes/FusedConvolutionBatchNormalizationNode.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/utils/misc/Cast.h"

#include <set>

namespace arm_compute
{
namespace graph
{
namespace detail
{
void fuse_convolution_with_batch_normalization(Graph &g, const Edge *output_edge)
{
    ARM_COMPUTE_ERROR_ON(output_edge == nullptr);

    auto *conv_node = arm_compute::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(output_edge->producer());
    auto *bn_node   = arm_compute::utils::cast::polymorphic_downcast<BatchNormalizationLayerNode *>(output_edge->consumer());

    // Not fusing if number of groups is greater than 1
    if(conv_node->num_groups() > 1)
    {
        return;
    }

    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Fusing convolution node with ID : " << output_edge->producer_id()
                                  << " with BatchNormalization Layer node with ID : " << output_edge->consumer_id() << std::endl);

    // Prevent fusion if fused node has an output accessor
    if(conv_node->output(0)->accessor() == nullptr)
    {
        const Target assigned_target = conv_node->assigned_target();

        // Extract conv inputs
        const auto   conv_input_id   = conv_node->input_edge(0)->producer_id();
        const auto   conv_weights_id = conv_node->input_edge(1)->producer_id();
        const auto   conv_info       = conv_node->convolution_info();
        const auto   conv_method     = conv_node->convolution_method();
        const auto   num_groups      = conv_node->num_groups();
        const auto   act_info        = bn_node->fused_activation();
        FastMathHint fast_math_hint  = conv_node->fast_math_hint();

        // Extract bn inputs
        const auto bn_mean_id  = bn_node->input_edge(1)->producer_id();
        const auto bn_var_id   = bn_node->input_edge(2)->producer_id();
        const auto bn_beta_id  = bn_node->input_edge(3)->producer_id();
        const auto bn_gamma_id = bn_node->input_edge(4)->producer_id();
        const auto epsilon     = bn_node->epsilon();

        // Create the fused node
        const NodeID fused_id = g.add_node<FusedConvolutionBatchNormalizationNode>(epsilon, conv_info, num_groups, conv_method, fast_math_hint, act_info);

        if(conv_node->input_edge(2) != nullptr)
        {
            auto conv_bias_id = conv_node->input_edge(2)->producer_id();
            g.add_connection(conv_bias_id, 0, fused_id, 2);
        }

        // Add connections from the conv/batch_norm inputs to the fused node
        g.add_connection(conv_input_id, 0, fused_id, 0);
        g.add_connection(conv_weights_id, 0, fused_id, 1);
        g.add_connection(bn_mean_id, 0, fused_id, 3);
        g.add_connection(bn_var_id, 0, fused_id, 4);
        g.add_connection(bn_beta_id, 0, fused_id, 5);
        g.add_connection(bn_gamma_id, 0, fused_id, 6);

        auto                     fused_node       = g.node(fused_id);
        std::vector<NodeIdxPair> bn_driving_nodes = get_driving_nodes(*bn_node);

        // Extract batch normalization node accessor if any
        auto bn_node_accessor = bn_node->output(0)->extract_accessor();
        auto bn_node_name     = bn_node->name();

        // Remove batch normalization node
        g.remove_node(bn_node->id());

        // Get driving nodes of batch normalization node
        for(auto &driving_node : bn_driving_nodes)
        {
            g.add_connection(fused_id, 0, driving_node.node_id, driving_node.index);
            configure_tensor(fused_node->output(0));
        }
        // Update fused node outputs
        fused_node->output(0)->set_accessor(std::move(bn_node_accessor));
        fused_node->set_assigned_target(assigned_target);
        fused_node->set_common_node_parameters(NodeParams{ conv_node->name() + "+" + bn_node_name, assigned_target });

        // Remove convolution node
        g.remove_node(conv_node->id());
    }
    else
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution with batch normalization due to the presence of an output accessor\n");
    }
}

void fuse_depthwise_convolution_with_batch_normalization(Graph &g, const Edge *output_edge)
{
    ARM_COMPUTE_ERROR_ON(output_edge == nullptr);

    auto *depth_conv_node = arm_compute::utils::cast::polymorphic_downcast<DepthwiseConvolutionLayerNode *>(output_edge->producer());
    auto *bn_node         = arm_compute::utils::cast::polymorphic_downcast<BatchNormalizationLayerNode *>(output_edge->consumer());

    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Fusing depthwise convolution node with ID : " << output_edge->producer_id()
                                  << " with BatchNormalization Layer node with ID : " << output_edge->consumer_id() << std::endl);

    // Prevent fusion if fused node has an output accessor
    if(depth_conv_node->output(0)->accessor() == nullptr)
    {
        const Target assigned_target = depth_conv_node->assigned_target();

        // Extract conv inputs
        const auto depth_conv_input_id = depth_conv_node->input_edge(0)->producer_id();
        const auto conv_weights_id     = depth_conv_node->input_edge(1)->producer_id();
        const auto conv_info           = depth_conv_node->convolution_info();
        const auto depth_conv_method   = depth_conv_node->depthwise_convolution_method();
        const auto depth_multiplier    = depth_conv_node->depth_multiplier();
        const auto act_info            = bn_node->fused_activation();

        // Extract bn inputs
        const auto bn_mean_id  = bn_node->input_edge(1)->producer_id();
        const auto bn_var_id   = bn_node->input_edge(2)->producer_id();
        const auto bn_beta_id  = bn_node->input_edge(3)->producer_id();
        const auto bn_gamma_id = bn_node->input_edge(4)->producer_id();
        const auto epsilon     = bn_node->epsilon();

        // Create the fused node
        const NodeID fused_id = g.add_node<FusedDepthwiseConvolutionBatchNormalizationNode>(epsilon, conv_info, depth_multiplier, depth_conv_method, act_info);

        if(depth_conv_node->input_edge(2) != nullptr)
        {
            const auto conv_bias_id = depth_conv_node->input_edge(2)->producer_id();
            g.add_connection(conv_bias_id, 0, fused_id, 2);
        }

        // Add connections from the conv/batch_norm inputs to the fused node
        g.add_connection(depth_conv_input_id, 0, fused_id, 0);
        g.add_connection(conv_weights_id, 0, fused_id, 1);
        g.add_connection(bn_mean_id, 0, fused_id, 3);
        g.add_connection(bn_var_id, 0, fused_id, 4);
        g.add_connection(bn_beta_id, 0, fused_id, 5);
        g.add_connection(bn_gamma_id, 0, fused_id, 6);

        auto                     fused_node       = g.node(fused_id);
        std::vector<NodeIdxPair> bn_driving_nodes = get_driving_nodes(*bn_node);

        // Extract batch normalization node accessor if any
        auto bn_node_accessor = bn_node->output(0)->extract_accessor();
        auto bn_node_name     = bn_node->name();

        // Remove batch normalization node
        g.remove_node(bn_node->id());

        // Get driving nodes of batch normalization node
        for(auto &driving_node : bn_driving_nodes)
        {
            g.add_connection(fused_id, 0, driving_node.node_id, driving_node.index);
            configure_tensor(fused_node->output(0));
        }
        // Update fused node outputs
        fused_node->output(0)->set_accessor(std::move(bn_node_accessor));
        fused_node->set_assigned_target(assigned_target);
        fused_node->set_common_node_parameters(NodeParams{ depth_conv_node->name() + "+" + bn_node_name, assigned_target });

        // Remove convolution node
        g.remove_node(depth_conv_node->id());
    }
    else
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of depthwise convolution with batch normalization due to the presence of an output accessor\n");
    }
}

template <typename N>
void fuse_node_with_activation(Graph &g, const Edge *output_edge, const std::set<Activation> &supported_fused_activations)
{
    ARM_COMPUTE_ERROR_ON(output_edge == nullptr);

    auto *n_node   = arm_compute::utils::cast::polymorphic_downcast<N *>(output_edge->producer());
    auto *act_node = arm_compute::utils::cast::polymorphic_downcast<ActivationLayerNode *>(output_edge->consumer());

    ARM_COMPUTE_ERROR_ON(act_node->output(0) == nullptr || n_node->output(0) == nullptr);

    // Check if activation is supported for fusion
    if(supported_fused_activations.count(act_node->activation_info().activation()) == 0)
    {
        return;
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

template <typename N1, typename N2, typename F, typename... Args>
void fuse_layer(Graph &g, std::function<bool(INode &)> const &prec, const F fuse_fcn, Args &&... optional_arguments)
{
    // Not interested in the order of nodes
    for(auto &node : g.nodes())
    {
        // Check if the node is of type N and not a branching node
        if(node && node->type() == N1::node_type && node->output_edges().size() == 1)
        {
            const auto output_edge_id = *node->output_edges().begin();
            const auto output_edge    = g.edge(output_edge_id);

            // Check if following node is an activation layer node
            if((output_edge != nullptr) && (output_edge->consumer() != nullptr) && (output_edge->consumer()->type() == N2::node_type) && prec(*output_edge->producer()))
            {
                fuse_fcn(g, output_edge, optional_arguments...);
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
    auto empty_prec = [](INode &)
    {
        return true;
    };
    auto qs8_prec = [&g](INode & n)
    {
        ARM_COMPUTE_ERROR_ON(n.output(0) == nullptr);

        const auto output_edge_id = *n.output_edges().begin();
        const auto output_edge    = g.edge(output_edge_id);
        // To perform fusion the two nodes must have same output quantization information
        const bool same_qinfo     = n.output(0)->desc().quant_info == output_edge->producer()->output(0)->desc().quant_info;
        const bool output_qasymm8 = n.output(0)->desc().data_type == DataType::QASYMM8;

        return (output_qasymm8 && same_qinfo) || !output_qasymm8;
    };

    Target target = g.nodes()[0].get()->output(0)->desc().target;

    // Fusion mutations
    detail::fuse_layer<BatchNormalizationLayerNode, ActivationLayerNode>(g, empty_prec, detail::fuse_node_with_activation<BatchNormalizationLayerNode>, supported_fused_activations);
    detail::fuse_layer<ConvolutionLayerNode, ActivationLayerNode>(g, empty_prec, detail::fuse_node_with_activation<ConvolutionLayerNode>, supported_fused_activations);
    detail::fuse_layer<DepthwiseConvolutionLayerNode, ActivationLayerNode>(g, qs8_prec, detail::fuse_node_with_activation<DepthwiseConvolutionLayerNode>, supported_fused_activations);

    // Currently fuse batch normalization brings performance uplift only on OpenCL with FP32 data type
    // TODO (COMPMID-2524): Fuse batch normalization with convolution and depthwise convolution at graph level for NEON - FP32
    // TODO (COMPMID-2581): Fuse batch normalization with convolution and depthwise convolution at graph level for OpenCL - FP16
    if(target == Target::CL && (g.nodes()[0].get()->output(0)->desc().data_type == DataType::F32))
    {
        //Depthwise Convolution and Batch Normalization Fusion active only for CL
        detail::fuse_layer<ConvolutionLayerNode, BatchNormalizationLayerNode>(g, empty_prec, detail::fuse_convolution_with_batch_normalization);
        detail::fuse_layer<DepthwiseConvolutionLayerNode, BatchNormalizationLayerNode>(g, empty_prec, detail::fuse_depthwise_convolution_with_batch_normalization);
    }
}
} // namespace graph
} // namespace arm_compute
