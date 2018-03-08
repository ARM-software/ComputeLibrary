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
#include "arm_compute/graph2/GraphBuilder.h"

#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/Utils.h"
#include "arm_compute/graph2/algorithms/BFS.h"
#include "arm_compute/graph2/nodes/Nodes.h"

#define CHECK_NODEIDX_PAIR(pair, g) \
    ARM_COMPUTE_ERROR_ON(((pair).node_id >= (g).nodes().size()) || ((g).node((pair).node_id) == nullptr) || ((pair).index >= (g).node((pair).node_id)->num_outputs()));

namespace arm_compute
{
namespace graph2
{
namespace
{
Status set_node_params(Graph &g, NodeID nid, NodeParams &params)
{
    INode *node = g.node(nid);
    ARM_COMPUTE_RETURN_ERROR_ON(!node);

    node->set_common_node_parameters(params);

    return Status{};
}

Status set_accessor_on_node(Graph &g, NodeID nid, bool is_output, size_t idx, ITensorAccessorUPtr accessor)
{
    INode *node = g.node(nid);
    ARM_COMPUTE_RETURN_ERROR_ON(!node);

    Tensor *tensor = is_output ? node->output(idx) : node->input(idx);
    ARM_COMPUTE_RETURN_ERROR_ON(!tensor);

    tensor->set_accessor(std::move(accessor));

    return Status{};
}

NodeID add_const_node_with_name(Graph &g, NodeParams params, const std::string &name, TensorDescriptor desc, ITensorAccessorUPtr accessor)
{
    params.name = params.name.empty() ? "" : params.name + name;
    auto nid    = GraphBuilder::add_const_node(g, params, desc, std::move(accessor));
    set_node_params(g, nid, params);
    return nid;
}

template <typename NT, typename... Args>
NodeID create_simple_single_input_output_node(Graph &g, NodeParams &params, NodeIdxPair input, Args &&... args)
{
    CHECK_NODEIDX_PAIR(input, g);

    NodeID nid = g.add_node<NT>(std::forward<Args>(args)...);
    g.add_connection(input.node_id, input.index, nid, 0);
    set_node_params(g, nid, params);

    return nid;
}

NodeID create_grouped_convolution(Graph &g, NodeParams &params, NodeIdxPair input, NodeID weights, NodeID bias,
                                  PadStrideInfo conv_info, ConvolutionMethod method, unsigned int num_groups)
{
    bool has_bias = (bias != EmptyNodeID);

    // Split input
    NodeID input_split = GraphBuilder::add_split_node(g, params, input, num_groups, 2);

    // Split weights
    NodeID weights_split = GraphBuilder::add_split_node(g, params, { weights, 0 }, num_groups, 3);

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
        NodeID conv_nid = g.add_node<ConvolutionLayerNode>(conv_info, method);
        g.add_connection(input_split, i, conv_nid, 0);
        g.add_connection(weights_split, i, conv_nid, 1);
        if(has_bias)
        {
            g.add_connection(bias_split, i, conv_nid, 2);
        }
        set_node_params(g, conv_nid, params);
        convolution_outputs.push_back({ conv_nid, 0 });
    }

    // Depth concatenate output
    return GraphBuilder::add_depth_concatenate_node(g, params, convolution_outputs);
}
} // namespace

NodeID GraphBuilder::add_const_node(Graph &g, NodeParams params, TensorDescriptor desc, ITensorAccessorUPtr accessor)
{
    auto nid = g.add_node<ConstNode>(desc);
    set_node_params(g, nid, params);
    set_accessor_on_node(g, nid, true, 0, std::move(accessor));
    return nid;
}

NodeID GraphBuilder::add_input_node(Graph &g, NodeParams params, TensorDescriptor desc, ITensorAccessorUPtr accessor)
{
    auto nid = g.add_node<InputNode>(desc);
    set_node_params(g, nid, params);
    set_accessor_on_node(g, nid, true, 0, std::move(accessor));
    return nid;
}

NodeID GraphBuilder::add_output_node(Graph &g, NodeParams params, NodeIdxPair input, ITensorAccessorUPtr accessor)
{
    CHECK_NODEIDX_PAIR(input, g);

    NodeID nid = g.add_node<OutputNode>();
    g.add_connection(input.node_id, input.index, nid, 0);
    set_node_params(g, nid, params);
    set_accessor_on_node(g, nid, false, 0, std::move(accessor));

    return nid;
}

NodeID GraphBuilder::add_activation_node(Graph &g, NodeParams params, NodeIdxPair input, ActivationLayerInfo act_info)
{
    return create_simple_single_input_output_node<ActivationLayerNode>(g, params, input, act_info);
}

NodeID GraphBuilder::add_batch_normalization_node(Graph &g, NodeParams params, NodeIdxPair input, float epsilon,
                                                  ITensorAccessorUPtr mean_accessor, ITensorAccessorUPtr var_accessor,
                                                  ITensorAccessorUPtr beta_accessor, ITensorAccessorUPtr gamma_accessor)
{
    CHECK_NODEIDX_PAIR(input, g);

    bool has_beta  = (beta_accessor != nullptr);
    bool has_gamma = (gamma_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Calculate Common Descriptor
    TensorDescriptor common_desc = input_tensor_desc;
    common_desc.shape            = TensorShape(common_desc.shape.z());

    // Create mean and nodes
    auto mean_nid = add_const_node_with_name(g, params, "Mean", common_desc, std::move(mean_accessor));
    auto var_nid  = add_const_node_with_name(g, params, "Variance", common_desc, std::move(var_accessor));

    // Create beta node
    NodeID beta_nid = EmptyNodeID;
    if(has_beta)
    {
        beta_nid = add_const_node_with_name(g, params, "Beta", common_desc, std::move(beta_accessor));
    }

    // Create gamma node
    NodeID gamma_nid = EmptyNodeID;
    if(has_gamma)
    {
        gamma_nid = add_const_node_with_name(g, params, "Gamma", common_desc, std::move(gamma_accessor));
    }

    // Create batch normalization node and add connections
    NodeID batch_norm_nid = g.add_node<BatchNormalizationLayerNode>(epsilon);
    g.add_connection(input.node_id, input.index, batch_norm_nid, 0);
    g.add_connection(mean_nid, 0, batch_norm_nid, 1);
    g.add_connection(var_nid, 0, batch_norm_nid, 2);
    if(has_beta)
    {
        g.add_connection(beta_nid, 0, batch_norm_nid, 3);
    }
    if(has_gamma)
    {
        g.add_connection(gamma_nid, 0, batch_norm_nid, 4);
    }
    set_node_params(g, batch_norm_nid, params);

    return batch_norm_nid;
}

NodeID GraphBuilder::add_convolution_node(Graph &g, NodeParams params, NodeIdxPair input,
                                          Size2D kernel_spatial_extend, unsigned int depth, PadStrideInfo conv_info,
                                          unsigned int num_groups, ConvolutionMethod method,
                                          ITensorAccessorUPtr weights_accessor, ITensorAccessorUPtr bias_accessor)
{
    CHECK_NODEIDX_PAIR(input, g);
    ARM_COMPUTE_ERROR_ON(depth == 0);
    ARM_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

    bool has_bias = (bias_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Create weights node
    TensorDescriptor w_desc = input_tensor_desc;
    w_desc.shape            = TensorShape(kernel_spatial_extend.width, kernel_spatial_extend.height, w_desc.shape.z() / num_groups, depth);
    NodeID w_nid            = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

    // Create bias nodes
    NodeID b_nid = EmptyNodeID;
    if(has_bias)
    {
        TensorDescriptor b_desc = input_tensor_desc;
        b_desc.shape            = TensorShape(depth);
        b_nid                   = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
    }

    if(num_groups == 1)
    {
        // Create convolution node and connect
        NodeID conv_nid = g.add_node<ConvolutionLayerNode>(conv_info, method);
        g.add_connection(input.node_id, input.index, conv_nid, 0);
        g.add_connection(w_nid, 0, conv_nid, 1);
        if(has_bias)
        {
            g.add_connection(b_nid, 0, conv_nid, 2);
        }
        set_node_params(g, conv_nid, params);

        return conv_nid;
    }
    else
    {
        return create_grouped_convolution(g, params, input, w_nid, b_nid, conv_info, method, num_groups);
    }
}

NodeID GraphBuilder::add_depth_concatenate_node(Graph &g, NodeParams params, std::vector<NodeIdxPair> inputs)
{
    ARM_COMPUTE_ERROR_ON(inputs.size() == 0);

    NodeID nid = g.add_node<DepthConcatenateLayerNode>(inputs.size());

    unsigned int i = 0;
    for(const auto &input : inputs)
    {
        CHECK_NODEIDX_PAIR(input, g);
        g.add_connection(input.node_id, input.index, nid, i++);
    }
    set_node_params(g, nid, params);

    return nid;
}

NodeID GraphBuilder::add_depthwise_convolution_node(Graph &g, NodeParams params, NodeIdxPair input, Size2D kernel_spatial_extend, PadStrideInfo conv_info,
                                                    DepthwiseConvolutionMethod method,
                                                    ITensorAccessorUPtr weights_accessor, ITensorAccessorUPtr bias_accessor)
{
    CHECK_NODEIDX_PAIR(input, g);
    ARM_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

    bool has_bias = (bias_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Create weights node
    TensorDescriptor w_desc = input_tensor_desc;
    w_desc.shape            = TensorShape(kernel_spatial_extend.width, kernel_spatial_extend.height, w_desc.shape.z());
    NodeID w_nid            = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

    // Create bias nodes
    NodeID b_nid = EmptyNodeID;
    if(has_bias)
    {
        TensorDescriptor b_desc = input_tensor_desc;
        b_desc.shape            = TensorShape(b_desc.shape.z());
        b_nid                   = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
    }

    // Create convolution node and connect
    NodeID conv_nid = g.add_node<DepthwiseConvolutionLayerNode>(conv_info, method);
    g.add_connection(input.node_id, input.index, conv_nid, 0);
    g.add_connection(w_nid, 0, conv_nid, 1);
    if(has_bias)
    {
        g.add_connection(b_nid, 0, conv_nid, 2);
    }
    set_node_params(g, conv_nid, params);

    return conv_nid;
}

NodeID GraphBuilder::add_elementwise_node(Graph &g, NodeParams params, NodeIdxPair input0, NodeIdxPair input1, EltwiseOperation operation)
{
    CHECK_NODEIDX_PAIR(input0, g);
    CHECK_NODEIDX_PAIR(input1, g);

    NodeID nid = g.add_node<EltwiseLayerNode>(operation);

    g.add_connection(input0.node_id, input0.index, nid, 0);
    g.add_connection(input1.node_id, input1.index, nid, 1);

    set_node_params(g, nid, params);

    return nid;
}

NodeID GraphBuilder::add_flatten_node(Graph &g, NodeParams params, NodeIdxPair input)
{
    return create_simple_single_input_output_node<FlattenLayerNode>(g, params, input);
}

NodeID GraphBuilder::add_fully_connected_layer(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_outputs,
                                               ITensorAccessorUPtr weights_accessor, ITensorAccessorUPtr bias_accessor)
{
    CHECK_NODEIDX_PAIR(input, g);
    ARM_COMPUTE_ERROR_ON(num_outputs == 0);

    bool has_bias = (bias_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Create weights node
    TensorDescriptor w_desc = input_tensor_desc;
    w_desc.shape            = FullyConnectedLayerNode::compute_weights_shape(input_tensor_desc.shape, num_outputs);
    NodeID w_nid            = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

    // Create bias nodes
    NodeID b_nid = EmptyNodeID;
    if(has_bias)
    {
        TensorDescriptor b_desc = input_tensor_desc;
        b_desc.shape            = TensorShape(num_outputs);
        b_nid                   = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
    }

    // Create convolution node and connect
    NodeID fc_nid = g.add_node<FullyConnectedLayerNode>(num_outputs);
    g.add_connection(input.node_id, input.index, fc_nid, 0);
    g.add_connection(w_nid, 0, fc_nid, 1);
    if(has_bias)
    {
        g.add_connection(b_nid, 0, fc_nid, 2);
    }

    set_node_params(g, fc_nid, params);

    return fc_nid;
}

NodeID GraphBuilder::add_normalization_node(Graph &g, NodeParams params, NodeIdxPair input, NormalizationLayerInfo norm_info)
{
    return create_simple_single_input_output_node<NormalizationLayerNode>(g, params, input, norm_info);
}

NodeID GraphBuilder::add_pooling_node(Graph &g, NodeParams params, NodeIdxPair input, PoolingLayerInfo pool_info)
{
    return create_simple_single_input_output_node<PoolingLayerNode>(g, params, input, pool_info);
}

NodeID GraphBuilder::add_reshape_node(Graph &g, NodeParams params, NodeIdxPair input, TensorShape shape)
{
    return create_simple_single_input_output_node<ReshapeLayerNode>(g, params, input, shape);
}

NodeID GraphBuilder::add_softmax_node(Graph &g, NodeParams params, NodeIdxPair input, float beta)
{
    return create_simple_single_input_output_node<SoftmaxLayerNode>(g, params, input, beta);
}

NodeID GraphBuilder::add_split_node(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_splits, unsigned int axis)
{
    return create_simple_single_input_output_node<SplitLayerNode>(g, params, input, num_splits, axis);
}
} // namespace graph2
} // namespace arm_compute