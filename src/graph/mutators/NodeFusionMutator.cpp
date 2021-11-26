/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/graph/nodes/FusedConvolutionBatchNormalizationWithPostOpsNode.h"
#include "arm_compute/graph/nodes/FusedConvolutionWithPostOpNode.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "src/graph/mutators/MutatorUtils.h"

#include "support/Cast.h"

#include <list>
#include <set>

namespace arm_compute
{
namespace graph
{
namespace detail
{
void transfer_driving_nodes_and_remove_old_node(Graph &g, INode *new_node, INode *old_node, bool add_output_tensor)
{
    if(new_node == nullptr || old_node == nullptr)
    {
        return;
    }

    // Get driving nodes of last fusable node
    std::vector<NodeIdxPair> last_driving_nodes = get_driving_nodes(*old_node);

    // Extract last fusable node accessor if any
    if(old_node->output(0) == nullptr)
    {
        return;
    }
    auto old_node_accessor = old_node->output(0)->extract_accessor();

    // Remove node
    g.remove_node(old_node->id());

    // Update fused node outputs
    for(auto &driving_node : last_driving_nodes)
    {
        g.add_connection(new_node->id(), 0, driving_node.node_id, driving_node.index);
        if(add_output_tensor)
        {
            configure_tensor(new_node->output(0));
        }
    }

    // Update accessor to fused node
    new_node->output(0)->set_accessor(std::move(old_node_accessor));
}

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
        const auto bn_mean_id = bn_node->input_edge(1)->producer_id();
        const auto bn_var_id  = bn_node->input_edge(2)->producer_id();

        const auto epsilon = bn_node->epsilon();

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

        if(bn_node->input_edge(3) != nullptr)
        {
            const auto bn_beta_id = bn_node->input_edge(3)->producer_id();
            g.add_connection(bn_beta_id, 0, fused_id, 5);
        }

        if(bn_node->input_edge(4) != nullptr)
        {
            const auto bn_gamma_id = bn_node->input_edge(4)->producer_id();
            g.add_connection(bn_gamma_id, 0, fused_id, 6);
        }

        auto fused_node   = g.node(fused_id);
        auto bn_node_name = bn_node->name();

        transfer_driving_nodes_and_remove_old_node(g, fused_node, bn_node, true);

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

        auto fused_node   = g.node(fused_id);
        auto bn_node_name = bn_node->name();

        transfer_driving_nodes_and_remove_old_node(g, fused_node, bn_node, true);

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

    // EltwiseLayerNode can only be fused when dataype is float
    if(n_node->type() == NodeType::EltwiseLayer && !is_data_type_float(n_node->output(0)->desc().data_type))
    {
        return;
    }

    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Fusing node with ID : " << output_edge->producer_id()
                                  << " with Activation Layer node with ID : " << output_edge->consumer_id() << std::endl);

    // Prevent fusion if fused node has an output accessor
    if(n_node->output(0)->accessor() == nullptr)
    {
        // Set activation info to fused node
        n_node->set_fused_activation(act_node->activation_info());

        transfer_driving_nodes_and_remove_old_node(g, n_node, act_node, false);
    }
    else
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of node with activation due to the presence of an output accessor\n");
    }
}

template <typename N>
void fuse_pad_with_convolution(Graph &g, const Edge *output_edge)
{
    auto *pad_node  = arm_compute::utils::cast::polymorphic_downcast<PadLayerNode *>(output_edge->producer());
    auto *conv_node = arm_compute::utils::cast::polymorphic_downcast<N *>(output_edge->consumer());

    const Edge *input_edge = pad_node->input_edge(0);
    if(input_edge != nullptr && input_edge->tensor() != nullptr && pad_node->output(0)->accessor() == nullptr
       && pad_node->pad_value().get<float>() == 0.0)
    {
        const DataLayout  layout       = input_edge->tensor()->desc().layout;
        const PaddingList padding_list = pad_node->padding();

        const unsigned int height_index = get_dimension_idx(layout, DataLayoutDimension::HEIGHT);
        const unsigned int width_index  = get_dimension_idx(layout, DataLayoutDimension::WIDTH);

        const PaddingInfo pad_w = width_index < padding_list.size() ? padding_list[width_index] : PaddingInfo(0, 0);
        const PaddingInfo pad_h = height_index < padding_list.size() ? padding_list[height_index] : PaddingInfo(0, 0);

        if(is_padding_in_height_or_width(layout, padding_list))
        {
            // Add paddings to the convolution node
            const PadStrideInfo conv_info = conv_node->convolution_info();
            const PadStrideInfo new_conv_info(
                conv_info.stride().first,
                conv_info.stride().second,
                conv_info.pad_left() + pad_w.first,
                conv_info.pad_right() + pad_w.second,
                conv_info.pad_top() + pad_h.first,
                conv_info.pad_bottom() + pad_h.second,
                conv_info.round());
            conv_node->set_convolution_info(new_conv_info);

            // Update drivers of the convolution node
            std::vector<NodeIdxPair> pad_driver_nodes = get_driver_nodes(*pad_node);
            g.remove_node(pad_node->id());

            // Update fused node inputs
            for(auto &driver_node : pad_driver_nodes)
            {
                g.add_connection(driver_node.node_id, driver_node.index, conv_node->id(), 0);
            }
        }
    }
}

template <typename N1, typename N2, typename F, typename... Args>
void fuse_layer(Graph &g, std::function<bool(INode &)> const &prec, const F fuse_fcn, Args &&... optional_arguments)
{
    // Note that fused nodes may be added to the end of the node list.
    // Instead of only looping over the original list of nodes, we loop over the current node list which could be growing.
    // This is intentional as it probes the newly added fused nodes for further fusing opportunities.
    for(unsigned int i = 0; i < g.nodes().size(); ++i)
    {
        auto node = g.node(i);
        // Check if the node is of type N1 and not a branching node
        if(node && node->type() == N1::node_type && node->output_edges().size() == 1)
        {
            const auto output_edge_id = *node->output_edges().begin();
            const auto output_edge    = g.edge(output_edge_id);

            // Check if following node is a type N2 node
            if((output_edge != nullptr) && (output_edge->consumer() != nullptr) && (output_edge->consumer()->type() == N2::node_type) && prec(*output_edge->producer()))
            {
                fuse_fcn(g, output_edge, optional_arguments...);
            }
        }
    }
}

/** Check valid combinations:
 *
 * | Main operator | Post operators             |
 * |:--------------|:---------------------------|
 * |conv           | add                        |
 * |conv           | act + add                  |
 * |conv           | add + act                  |
 * |conv           | act + add + act            |
 *
*/
#define MAX_VALIDE_COMBINATION 4
#define MAX_POST_OP_NUM 3
NodeType valide_post_op_type[MAX_VALIDE_COMBINATION][MAX_POST_OP_NUM] = { { EltwiseLayerNode::node_type },
    { EltwiseLayerNode::node_type, ActivationLayerNode::node_type },
    { ActivationLayerNode::node_type, EltwiseLayerNode::node_type },
    { ActivationLayerNode::node_type, EltwiseLayerNode::node_type, ActivationLayerNode::node_type }
};

bool check_post_op_type(NodeType *post_op_type, int len)
{
    if(len > MAX_POST_OP_NUM || len <= 0)
    {
        return false;
    }

    bool found = false;
    for(int i = 0; i < MAX_VALIDE_COMBINATION; ++i)
    {
        for(int j = 0; j < len; ++j)
        {
            if(post_op_type[j] != valide_post_op_type[i][j])
            {
                found = false;
                break;
            }
            found = true;
        }
        if(found)
            break;
    }

    return found;
}

void fuse_convolution_with_post_op(Graph &g, INode *fused_node, std::list<INode *> post_op_node_list, int prev_op_dst_pos)
{
    unsigned int op_idx = 0;
    // Fuse post operators with conv
    for(const auto &post_op : post_op_node_list)
    {
        switch(post_op->type())
        {
            case EltwiseLayerNode::node_type:
            {
                auto *eltwise_node = arm_compute::utils::cast::polymorphic_downcast<EltwiseLayerNode *>(post_op);
                ARM_COMPUTE_ERROR_ON(eltwise_node->output(0) == nullptr);

                fused_node->post_op_info_list().push_back(std::make_unique<ConvPostOpInfoEltwiseAdd>(prev_op_dst_pos, eltwise_node->convert_policy()));
                ARM_COMPUTE_LOG_GRAPH_VERBOSE(" with Elementwise Layer node with ID : " << post_op->id());
                break;
            }
            case ActivationLayerNode::node_type:
            {
                auto *act_node = arm_compute::utils::cast::polymorphic_downcast<ActivationLayerNode *>(post_op);
                ARM_COMPUTE_ERROR_ON(act_node->output(0) == nullptr);

                fused_node->post_op_info_list().push_back(std::make_unique<ConvPostOpInfoActivation>(act_node->activation_info()));
                ARM_COMPUTE_LOG_GRAPH_VERBOSE(" with Activation Layer node with ID : " << post_op->id());
                break;
            }
            default:
            {
                break;
            }
        }

        if(op_idx == post_op_node_list.size() - 1) // last fusable node
        {
            transfer_driving_nodes_and_remove_old_node(g, fused_node, post_op, true);
        }
        else
        {
            // Remove node
            g.remove_node(post_op->id());
        }
        op_idx++;
    }
}

std::list<INode *> get_post_op_list(Graph &g, int &eltwise_operand_id, int &prev_op_dst_pos, unsigned int conv_node_id, const std::set<Activation> &supported_fused_activations)
{
    std::list<INode *> post_op_node_list    = {};
    NodeID             prev_op_dst_id       = conv_node_id;
    NodeType           post_op_type_list[3] = { NodeType::Dummy, NodeType::Dummy, NodeType::Dummy };
    int                post_op_idx          = 0;

    // Get list of the connected nodes
    auto current_node = g.node(conv_node_id);

    while(post_op_node_list.size() < 3)
    {
        // This convolution node must have only one output edge, otherwise this function would not have been called

        auto current_output_edge_id = current_node->output_edges().begin();
        auto current_output_edge    = g.edge(*current_output_edge_id);
        auto post_op_node           = current_output_edge->consumer();

        bool fusable_post_op = false;
        if(post_op_node != nullptr && post_op_node->output_edges().size() > 0)
        {
            switch(post_op_node->type())
            {
                case EltwiseLayerNode::node_type:
                {
                    auto *eltwise_node = arm_compute::utils::cast::polymorphic_downcast<EltwiseLayerNode *>(post_op_node);
                    ARM_COMPUTE_ERROR_ON(eltwise_node->output(0) == nullptr);
                    if(eltwise_node->output(0)->accessor() == nullptr)
                    {
                        post_op_node_list.push_back(post_op_node);
                        fusable_post_op                  = true;
                        post_op_type_list[post_op_idx++] = eltwise_node->type();

                        // Extract elementwise inputs
                        const auto eltwise_input_id_0 = eltwise_node->input_edge(0)->producer_id();
                        const auto eltwise_input_id_1 = eltwise_node->input_edge(1)->producer_id();
                        if(eltwise_input_id_0 == prev_op_dst_id)
                        {
                            eltwise_operand_id = eltwise_input_id_1;
                            prev_op_dst_pos    = 0;
                        }
                        else if(eltwise_input_id_1 == prev_op_dst_id)
                        {
                            eltwise_operand_id = eltwise_input_id_0;
                            prev_op_dst_pos    = 1;
                        }
                    }
                    else
                    {
                        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with elementwise due to the presence of an output accessor\n");
                    }
                    break;
                }
                case ActivationLayerNode::node_type:
                {
                    auto *act_node = arm_compute::utils::cast::polymorphic_downcast<ActivationLayerNode *>(post_op_node);
                    ARM_COMPUTE_ERROR_ON(act_node->output(0) == nullptr);
                    // Check if activation is supported for fusion
                    if(supported_fused_activations.count(act_node->activation_info().activation()) == 0)
                    {
                        break;
                    }
                    if(act_node->output(0)->accessor() == nullptr)
                    {
                        post_op_node_list.push_back(post_op_node);
                        fusable_post_op                  = true;
                        post_op_type_list[post_op_idx++] = act_node->type();
                        prev_op_dst_id                   = act_node->id();
                    }
                    else
                    {
                        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with post ops due to the presence of an output accessor\n");
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }

            // Check if the node is not a branching node and current node is fusable
            if(post_op_node->output_edges().size() == 1 && fusable_post_op == true)
            {
                current_node = post_op_node;
            }
            else
            {
                break;
            }
        }
    }

    // Check whether it's valid post op list
    if(post_op_node_list.size() > 0)
    {
        bool fuse_with_post_op = check_post_op_type(post_op_type_list, post_op_node_list.size());
        if(!fuse_with_post_op)
        {
            post_op_node_list.clear();
        }
    }

    return post_op_node_list;
}

/** Fuse below operators:
 *
 * | Main operator | Post operators             |
 * |:--------------|:---------------------------|
 * |conv           | add                        |
 * |conv           | act + add                  |
 * |conv           | add + act                  |
 * |conv           | act + add + act            |
 *
 * Notes: currently, only GEMM supports fusion with post operator
*/
void fuse_convolution_with_post_ops(Graph &g, const Edge *output_edge, unsigned int conv_node_id, const std::set<Activation> &supported_fused_activations)
{
    ARM_COMPUTE_ERROR_ON(output_edge == nullptr);

    auto *conv_node = arm_compute::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(output_edge->producer());
    ARM_COMPUTE_ERROR_ON(conv_node->output(0) == nullptr);

    const ConvolutionMethod conv_algorithm = conv_node->convolution_method();
    if(conv_algorithm != ConvolutionMethod::GEMM)
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with post ops due to non GEMM convolution\n");
        return;
    }

    // Prevent fusion if fused node has an output accessor
    if(conv_node->output(0)->accessor() == nullptr)
    {
        // If data type is FP32/FP16, data layout is NHWC, and filter size is 1x1, fuse convolution with post op, as Conv1x1 always leads to GEMM.
        const Edge *input_edge = conv_node->input_edge(1);
        if(input_edge != nullptr && input_edge->tensor() != nullptr)
        {
            const DataLayout  data_layout  = input_edge->tensor()->desc().layout;
            const DataType    data_type    = input_edge->tensor()->desc().data_type;
            const TensorShape tensor_shape = input_edge->tensor()->desc().shape;
            if((data_layout != DataLayout::NHWC) || (is_data_type_float(data_type) == false) || (tensor_shape.y() != 1) || (tensor_shape.z() != 1))
            {
                ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with post ops due to non GEMM convolution\n");
                return;
            }
        }
        else
        {
            return;
        }

        // Get post op list
        int                eltwise_operand_id = 0;
        int                prev_op_dst_pos    = 0; // Previous operator dst's postion in current operator
        std::list<INode *> post_op_node_list  = get_post_op_list(g, eltwise_operand_id, prev_op_dst_pos, conv_node_id, supported_fused_activations);

        if(post_op_node_list.size() == 0)
        {
            return;
        }
        else // Do convolution fusion with post op if there're one(elementwise), two or more operators
        {
            const Target assigned_target = conv_node->assigned_target();

            // Extract conv inputs
            const auto   conv_input_id   = conv_node->input_edge(0)->producer_id();
            const auto   conv_weights_id = conv_node->input_edge(1)->producer_id();
            const auto   conv_info       = conv_node->convolution_info();
            const auto   conv_method     = conv_node->convolution_method();
            const auto   num_groups      = conv_node->num_groups();
            FastMathHint fast_math_hint  = conv_node->fast_math_hint();

            // Create the fused node
            const NodeID fused_id = g.add_node<FusedConvolutionWithPostOpNode>(conv_info, num_groups, conv_method, fast_math_hint);
            ARM_COMPUTE_LOG_GRAPH_VERBOSE("Fusing convolution node with ID : " << conv_node->id());

            // Add connections from the conv inputs to the fused node
            g.add_connection(conv_input_id, 0, fused_id, 0);
            g.add_connection(conv_weights_id, 0, fused_id, 1);
            if(conv_node->input_edge(2) != nullptr)
            {
                auto conv_bias_id = conv_node->input_edge(2)->producer_id();
                g.add_connection(conv_bias_id, 0, fused_id, 2);
            }
            // Adding the Element wise operand in case the post op is element wise operation
            auto it = std::find_if(post_op_node_list.begin(),
                                   post_op_node_list.end(),
                                   [&](const INode * nd)
            {
                return (nd->type() == graph::NodeType::EltwiseLayer);
            });

            if(it != post_op_node_list.end())
            {
                g.add_connection(eltwise_operand_id, 0, fused_id, 3);
            }
            g.remove_node(conv_node->id());

            // Update fused node outputs
            auto fused_node = g.node(fused_id);
            fused_node->set_assigned_target(assigned_target);

            // Fuse convolution with post op
            fuse_convolution_with_post_op(g, fused_node, post_op_node_list, prev_op_dst_pos);

            post_op_node_list.clear();
            ARM_COMPUTE_LOG_GRAPH_VERBOSE(std::endl);
        }
    }
    else
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with post ops due to the presence of an output accessor\n");
    }
}

void fuse_convolution_batch_normalization_with_post_ops(Graph &g, const Edge *output_edge, unsigned int conv_node_id, const std::set<Activation> &supported_fused_activations)
{
    ARM_COMPUTE_ERROR_ON(output_edge == nullptr);

    auto *conv_node = arm_compute::utils::cast::polymorphic_downcast<FusedConvolutionBatchNormalizationNode *>(output_edge->producer());
    ARM_COMPUTE_ERROR_ON(conv_node->output(0) == nullptr);
    const ConvolutionMethod conv_algorithm = conv_node->convolution_method();
    if(conv_algorithm != ConvolutionMethod::GEMM)
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with post ops due to non GEMM convolution\n");
        return;
    }

    // Prevent fusion if fused node has an output accessor
    if(conv_node->output(0)->accessor() == nullptr)
    {
        // If data type is FP32/FP16, data layout is NHWC, and filter size is 1x1, fuse convolution with post op, as Conv1x1 always leads to GEMM.
        const Edge *input_edge = conv_node->input_edge(1);
        if(input_edge != nullptr && input_edge->tensor() != nullptr)
        {
            const DataLayout  data_layout  = input_edge->tensor()->desc().layout;
            const DataType    data_type    = input_edge->tensor()->desc().data_type;
            const TensorShape tensor_shape = input_edge->tensor()->desc().shape;
            if((data_layout != DataLayout::NHWC) || (is_data_type_float(data_type) == false) || (tensor_shape.y() != 1) || (tensor_shape.z() != 1))
            {
                ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with post ops due to non GEMM convolution\n");
                return;
            }
        }
        else
        {
            return;
        }

        // Get post op list
        int                eltwise_operand_id = 0;
        int                prev_op_dst_pos    = 0; // Previous operator dst's postion in current operator
        std::list<INode *> post_op_node_list  = get_post_op_list(g, eltwise_operand_id, prev_op_dst_pos, conv_node_id, supported_fused_activations);

        if(post_op_node_list.size() == 0)
        {
            return;
        }
        else // Do convolution fusion with post op if there're one(elementwise), two or more operators
        {
            const Target assigned_target = conv_node->assigned_target();

            // Extract conv inputs
            const auto   conv_input_id   = conv_node->input_edge(0)->producer_id();
            const auto   conv_weights_id = conv_node->input_edge(1)->producer_id();
            const auto   bn_mean_id      = conv_node->input_edge(3)->producer_id();
            const auto   bn_var_id       = conv_node->input_edge(4)->producer_id();
            const auto   conv_info       = conv_node->convolution_info();
            const auto   conv_method     = conv_node->convolution_method();
            const auto   num_groups      = conv_node->num_groups();
            FastMathHint fast_math_hint  = conv_node->fast_math_hint();

            // Create the fused node

            const float  epsilon  = conv_node->epsilon();
            const NodeID fused_id = g.add_node<FusedConvolutionBatchNormalizationWithPostOpsNode>(epsilon, conv_info, num_groups, conv_method, fast_math_hint);

            ARM_COMPUTE_LOG_GRAPH_VERBOSE("Fusing FusedConvolutionBatchNormalization node with ID : " << conv_node->id());

            // Add connections from the conv inputs to the fused node
            g.add_connection(conv_input_id, 0, fused_id, 0);
            g.add_connection(conv_weights_id, 0, fused_id, 1);

            if(conv_node->input_edge(2) != nullptr)
            {
                auto conv_bias_id = conv_node->input_edge(2)->producer_id();
                g.add_connection(conv_bias_id, 0, fused_id, 2);
            }
            g.add_connection(bn_mean_id, 0, fused_id, 3);
            g.add_connection(bn_var_id, 0, fused_id, 4);

            // Move connections of old FusedConvolutionBatchNormalization to the fused node
            if(conv_node->input_edge(5) != nullptr)
            {
                const auto bn_beta_id = conv_node->input_edge(5)->producer_id();
                g.add_connection(bn_beta_id, 0, fused_id, 5);
            }

            if(conv_node->input_edge(6) != nullptr)
            {
                const auto bn_gamma_id = conv_node->input_edge(6)->producer_id();
                g.add_connection(bn_gamma_id, 0, fused_id, 6);
            }

            // Adding the Element wise operand in case the post op is element wise operation
            auto it = std::find_if(post_op_node_list.begin(),
                                   post_op_node_list.end(),
                                   [&](const INode * nd)
            {
                return (nd->type() == graph::NodeType::EltwiseLayer);
            });

            if(it != post_op_node_list.end())
            {
                g.add_connection(eltwise_operand_id, 0, fused_id, 7);
            }

            // Update fused node outputs
            auto fused_node = g.node(fused_id);
            fused_node->set_assigned_target(assigned_target);

            auto conv_node_name = conv_node->name();

            // collect the post ops names
            std::string post_ops_name = "";
            for(auto &post_op : post_op_node_list)
            {
                post_ops_name += post_op->name();
            }
            fused_node->set_common_node_parameters(NodeParams{ conv_node->name() + "+" + post_ops_name, assigned_target });

            // Fuse convolution with post op
            fuse_convolution_with_post_op(g, fused_node, post_op_node_list, prev_op_dst_pos);

            post_op_node_list.clear();
            g.remove_node(conv_node->id());
            ARM_COMPUTE_LOG_GRAPH_VERBOSE(std::endl);
        }
    }
    else
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Prevented fusion of convolution node with post ops due to the presence of an output accessor\n");
    }
}

template <typename N1, typename F, typename... Args>
void fuse_layer(Graph &g, std::function<bool(INode &)> const &prec, const F fuse_fcn, Args &&... optional_arguments)
{
    // Note that fused nodes may be added to the end of the node list.
    // Instead of only looping over the original list of nodes, we loop over the current node list which could be growing.
    // This is intentional as it probes the newly added fused nodes for further fusing opportunities.
    for(unsigned int i = 0; i < g.nodes().size(); ++i)
    {
        auto node = g.node(i);
        // Check if the node is of type N1 and not a branching node
        if(node && node->type() == N1::node_type && node->output_edges().size() == 1)
        {
            const auto output_edge_id = *node->output_edges().begin();
            const auto output_edge    = g.edge(output_edge_id);

            // Check if it's the correct target
            if((output_edge != nullptr) && (output_edge->consumer() != nullptr) && prec(*output_edge->producer()))
            {
                fuse_fcn(g, output_edge, i, optional_arguments...);
            }
        }
    }
}
} // namespace detail

const char *NodeFusionMutator::name()
{
    return "NodeFusionMutator";
}

IGraphMutator::MutationType NodeFusionMutator::type() const
{
    return IGraphMutator::MutationType::Backend;
}

void NodeFusionMutator::mutate(Graph &g)
{
    // Supported activations when fusing
    const std::set<Activation> supported_fused_activations = { Activation::ABS, Activation::BOUNDED_RELU, Activation::ELU,
                                                               Activation::HARD_SWISH, Activation::IDENTITY, Activation::LEAKY_RELU,
                                                               Activation::LINEAR, Activation::LOGISTIC, Activation::LU_BOUNDED_RELU,
                                                               Activation::RELU, Activation::SOFT_RELU, Activation::SQRT,
                                                               Activation::SQUARE, Activation::TANH
                                                             };

    // Preconditions
    auto empty_prec = [](INode &)
    {
        return true;
    };
    auto cl_target_prec = [](INode & n)
    {
        return n.assigned_target() == Target::CL;
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

    // Fusion mutations

    detail::fuse_layer<PadLayerNode, ConvolutionLayerNode>(g, empty_prec, detail::fuse_pad_with_convolution<ConvolutionLayerNode>);
    detail::fuse_layer<PadLayerNode, DepthwiseConvolutionLayerNode>(g, empty_prec, detail::fuse_pad_with_convolution<DepthwiseConvolutionLayerNode>);
    // The fusion of PostOps to ConvolutionLayer:
    // It must occur after the fusion of PadLayer into ConvolutionLayer
    // It must occur before the fusion of normal ActivationLayer into ConvolutionLayer as it takes precedence
    detail::fuse_layer<ConvolutionLayerNode>(g, cl_target_prec, detail::fuse_convolution_with_post_ops, supported_fused_activations);
    detail::fuse_layer<BatchNormalizationLayerNode, ActivationLayerNode>(g, empty_prec, detail::fuse_node_with_activation<BatchNormalizationLayerNode>, supported_fused_activations);
    detail::fuse_layer<ConvolutionLayerNode, ActivationLayerNode>(g, empty_prec, detail::fuse_node_with_activation<ConvolutionLayerNode>, supported_fused_activations);
    detail::fuse_layer<DepthwiseConvolutionLayerNode, ActivationLayerNode>(g, qs8_prec, detail::fuse_node_with_activation<DepthwiseConvolutionLayerNode>, supported_fused_activations);
    detail::fuse_layer<FullyConnectedLayerNode, ActivationLayerNode>(g, empty_prec, detail::fuse_node_with_activation<FullyConnectedLayerNode>, supported_fused_activations);
    detail::fuse_layer<EltwiseLayerNode, ActivationLayerNode>(g, cl_target_prec, detail::fuse_node_with_activation<EltwiseLayerNode>, supported_fused_activations);
    // The fusion of BatchNormalizationLayer must occur after the fusion of ActivationLayer. Because FusedConvolutionBatchNormalizationNode assumes the BatchNormalization is already fused with activation, if any
    detail::fuse_layer<ConvolutionLayerNode, BatchNormalizationLayerNode>(g, empty_prec, detail::fuse_convolution_with_batch_normalization);
    detail::fuse_layer<DepthwiseConvolutionLayerNode, BatchNormalizationLayerNode>(g, empty_prec, detail::fuse_depthwise_convolution_with_batch_normalization);
    detail::fuse_layer<FusedConvolutionBatchNormalizationNode>(g, cl_target_prec, detail::fuse_convolution_batch_normalization_with_post_ops, supported_fused_activations);
}
} // namespace graph
} // namespace arm_compute
