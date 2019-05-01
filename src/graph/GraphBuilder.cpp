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
#include "arm_compute/graph/GraphBuilder.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/algorithms/TopologicalSort.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace graph
{
namespace
{
inline void check_nodeidx_pair(const NodeIdxPair &pair, const Graph &g)
{
    ARM_COMPUTE_UNUSED(pair);
    ARM_COMPUTE_UNUSED(g);
    ARM_COMPUTE_ERROR_ON((pair.node_id >= g.nodes().size()) || (g.node((pair).node_id) == nullptr) || (pair.index >= g.node(pair.node_id)->num_outputs()));
}

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

NodeID add_const_node_with_name(Graph &g, NodeParams params, const std::string &name, const TensorDescriptor &desc, ITensorAccessorUPtr accessor)
{
    params.name = params.name.empty() ? "" : params.name + name;
    auto nid    = GraphBuilder::add_const_node(g, params, desc, std::move(accessor));
    set_node_params(g, nid, params);
    return nid;
}

template <typename NT, typename... Args>
NodeID create_simple_single_input_output_node(Graph &g, NodeParams &params, NodeIdxPair input, Args &&... args)
{
    check_nodeidx_pair(input, g);

    NodeID nid = g.add_node<NT>(std::forward<Args>(args)...);
    g.add_connection(input.node_id, input.index, nid, 0);
    set_node_params(g, nid, params);

    return nid;
}

template <typename NT, typename... Args>
NodeID create_simple_multiple_input_single_output_node(Graph &g, NodeParams &params, const std::vector<NodeIdxPair> &inputs, Args &&... args)
{
    ARM_COMPUTE_ERROR_ON(inputs.size() == 0);

    NodeID nid = g.add_node<NT>(std::forward<Args>(args)...);

    unsigned int i = 0;
    for(const auto &input : inputs)
    {
        check_nodeidx_pair(input, g);
        g.add_connection(input.node_id, input.index, nid, i++);
    }
    set_node_params(g, nid, params);

    return nid;
}
} // namespace

NodeID GraphBuilder::add_const_node(Graph &g, NodeParams params, const TensorDescriptor &desc, ITensorAccessorUPtr accessor)
{
    auto nid = g.add_node<ConstNode>(desc);
    set_node_params(g, nid, params);
    set_accessor_on_node(g, nid, true, 0, std::move(accessor));
    return nid;
}

NodeID GraphBuilder::add_input_node(Graph &g, NodeParams params, const TensorDescriptor &desc, ITensorAccessorUPtr accessor)
{
    auto nid = g.add_node<InputNode>(desc);
    set_node_params(g, nid, params);
    set_accessor_on_node(g, nid, true, 0, std::move(accessor));
    return nid;
}

NodeID GraphBuilder::add_output_node(Graph &g, NodeParams params, NodeIdxPair input, ITensorAccessorUPtr accessor)
{
    check_nodeidx_pair(input, g);

    NodeID nid = g.add_node<OutputNode>();
    g.add_connection(input.node_id, input.index, nid, 0);
    set_node_params(g, nid, params);
    set_accessor_on_node(g, nid, false, 0, std::move(accessor));

    return nid;
}

NodeID GraphBuilder::add_activation_node(Graph &g, NodeParams params, NodeIdxPair input, ActivationLayerInfo act_info,
                                         const QuantizationInfo out_quant_info)
{
    return create_simple_single_input_output_node<ActivationLayerNode>(g, params, input, act_info, out_quant_info);
}

NodeID GraphBuilder::add_batch_normalization_node(Graph &g, NodeParams params, NodeIdxPair input, float epsilon,
                                                  ITensorAccessorUPtr mean_accessor, ITensorAccessorUPtr var_accessor,
                                                  ITensorAccessorUPtr beta_accessor, ITensorAccessorUPtr gamma_accessor)
{
    check_nodeidx_pair(input, g);

    bool has_beta  = (beta_accessor != nullptr);
    bool has_gamma = (gamma_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Calculate Common Descriptor
    TensorDescriptor common_desc = input_tensor_desc;
    common_desc.shape            = TensorShape(get_dimension_size(input_tensor_desc, DataLayoutDimension::CHANNEL));

    // Create mean and var nodes
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

NodeID GraphBuilder::add_bounding_box_transform_node(Graph &g, NodeParams params, NodeIdxPair input, NodeIdxPair deltas, BoundingBoxTransformInfo info)
{
    check_nodeidx_pair(input, g);
    check_nodeidx_pair(deltas, g);

    NodeID nid = g.add_node<BoundingBoxTransformLayerNode>(info);

    g.add_connection(input.node_id, input.index, nid, 0);
    g.add_connection(deltas.node_id, deltas.index, nid, 1);

    set_node_params(g, nid, params);
    return nid;
}

NodeID GraphBuilder::add_channel_shuffle_node(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_groups)
{
    return create_simple_single_input_output_node<ChannelShuffleLayerNode>(g, params, input, num_groups);
}

NodeID GraphBuilder::add_convolution_node(Graph &g, NodeParams params, NodeIdxPair input,
                                          Size2D kernel_spatial_extend, unsigned int depth, PadStrideInfo conv_info,
                                          unsigned int num_groups, ConvolutionMethod method, FastMathHint fast_math_hint,
                                          ITensorAccessorUPtr weights_accessor, ITensorAccessorUPtr bias_accessor,
                                          const QuantizationInfo weights_quant_info,
                                          const QuantizationInfo out_quant_info)
{
    check_nodeidx_pair(input, g);
    ARM_COMPUTE_ERROR_ON(depth == 0);
    ARM_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

    bool has_bias = (bias_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
    const DataLayout       input_data_layout = input_tensor_desc.layout;

    // Create weights node
    TensorDescriptor w_desc = input_tensor_desc;
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::WIDTH), kernel_spatial_extend.width);
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::HEIGHT), kernel_spatial_extend.height);
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::CHANNEL),
                     get_dimension_size(input_tensor_desc, DataLayoutDimension::CHANNEL) / num_groups);
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::BATCHES), depth);
    if(!weights_quant_info.empty())
    {
        w_desc.quant_info = weights_quant_info;
    }

    NodeID w_nid = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

    // Create bias nodes
    NodeID b_nid = EmptyNodeID;
    if(has_bias)
    {
        TensorDescriptor b_desc = input_tensor_desc;
        b_desc.shape            = TensorShape(depth);
        if(is_data_type_quantized_asymmetric(input_tensor_desc.data_type))
        {
            b_desc.data_type = DataType::S32;
        }
        b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
    }

    // Create convolution node and connect
    NodeID conv_nid = g.add_node<ConvolutionLayerNode>(conv_info, num_groups, method, fast_math_hint, out_quant_info);
    g.add_connection(input.node_id, input.index, conv_nid, 0);
    g.add_connection(w_nid, 0, conv_nid, 1);
    if(has_bias)
    {
        g.add_connection(b_nid, 0, conv_nid, 2);
    }
    set_node_params(g, conv_nid, params);

    return conv_nid;
}

NodeID GraphBuilder::add_deconvolution_node(Graph &g, NodeParams params, NodeIdxPair input,
                                            Size2D kernel_spatial_extend, unsigned int depth, PadStrideInfo deconv_info,
                                            Size2D inner_border, ITensorAccessorUPtr weights_accessor,
                                            ITensorAccessorUPtr bias_accessor)
{
    check_nodeidx_pair(input, g);
    ARM_COMPUTE_ERROR_ON(depth == 0);
    ARM_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

    bool has_bias = (bias_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
    const DataLayout       input_data_layout = input_tensor_desc.layout;

    // Create weights node
    TensorDescriptor w_desc = input_tensor_desc;
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::WIDTH), kernel_spatial_extend.width);
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::HEIGHT), kernel_spatial_extend.height);
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::CHANNEL),
                     get_dimension_size(input_tensor_desc, DataLayoutDimension::CHANNEL));
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::BATCHES), depth);

    NodeID w_nid = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

    // Create bias nodes
    NodeID b_nid = EmptyNodeID;
    if(has_bias)
    {
        TensorDescriptor b_desc = input_tensor_desc;
        b_desc.shape            = TensorShape(depth);
        if(is_data_type_quantized_asymmetric(input_tensor_desc.data_type))
        {
            b_desc.data_type = DataType::S32;
        }
        b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
    }

    // Create convolution node and connect
    NodeID deconv_nid = g.add_node<DeconvolutionLayerNode>(deconv_info, inner_border);
    g.add_connection(input.node_id, input.index, deconv_nid, 0);
    g.add_connection(w_nid, 0, deconv_nid, 1);
    if(has_bias)
    {
        g.add_connection(b_nid, 0, deconv_nid, 2);
    }
    set_node_params(g, deconv_nid, params);

    return deconv_nid;
}

NodeID GraphBuilder::add_concatenate_node(Graph &g, NodeParams params, const std::vector<NodeIdxPair> &inputs, descriptors::ConcatLayerDescriptor concat_descriptor)
{
    return create_simple_multiple_input_single_output_node<ConcatenateLayerNode>(g, params, inputs, inputs.size(), concat_descriptor);
}

NodeID GraphBuilder::add_depthwise_convolution_node(Graph &g, NodeParams params, NodeIdxPair input, Size2D kernel_spatial_extend,
                                                    PadStrideInfo conv_info, int depth_multiplier, DepthwiseConvolutionMethod method,
                                                    ITensorAccessorUPtr weights_accessor, ITensorAccessorUPtr bias_accessor, const QuantizationInfo quant_info, const QuantizationInfo out_quant_info)
{
    check_nodeidx_pair(input, g);
    ARM_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

    bool has_bias = (bias_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
    const DataLayout       input_data_layout = input_tensor_desc.layout;

    // Create weights node
    TensorDescriptor w_desc = input_tensor_desc;
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::WIDTH), kernel_spatial_extend.width);
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::HEIGHT), kernel_spatial_extend.height);
    w_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::CHANNEL),
                     get_dimension_size(input_tensor_desc, DataLayoutDimension::CHANNEL) * depth_multiplier);
    if(!quant_info.empty())
    {
        w_desc.quant_info = quant_info;
    }

    NodeID w_nid = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

    // Create bias nodes
    NodeID b_nid = EmptyNodeID;
    if(has_bias)
    {
        TensorDescriptor b_desc = input_tensor_desc;
        b_desc.shape            = TensorShape(get_dimension_size(input_tensor_desc, DataLayoutDimension::CHANNEL) * depth_multiplier);

        if(is_data_type_quantized_asymmetric(b_desc.data_type))
        {
            b_desc.data_type = DataType::S32;
        }

        b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
    }

    // Create convolution node and connect
    NodeID conv_nid = g.add_node<DepthwiseConvolutionLayerNode>(conv_info, depth_multiplier, method, out_quant_info);
    g.add_connection(input.node_id, input.index, conv_nid, 0);
    g.add_connection(w_nid, 0, conv_nid, 1);
    if(has_bias)
    {
        g.add_connection(b_nid, 0, conv_nid, 2);
    }
    set_node_params(g, conv_nid, params);

    return conv_nid;
}
NodeID GraphBuilder::add_detection_output_node(Graph &g, NodeParams params, NodeIdxPair input_loc, NodeIdxPair input_conf, NodeIdxPair input_priorbox, const DetectionOutputLayerInfo &detect_info)
{
    check_nodeidx_pair(input_loc, g);
    check_nodeidx_pair(input_conf, g);
    check_nodeidx_pair(input_priorbox, g);

    // Create detection_output node and connect
    NodeID detect_nid = g.add_node<DetectionOutputLayerNode>(detect_info);
    g.add_connection(input_loc.node_id, input_loc.index, detect_nid, 0);
    g.add_connection(input_conf.node_id, input_conf.index, detect_nid, 1);
    g.add_connection(input_priorbox.node_id, input_priorbox.index, detect_nid, 2);

    set_node_params(g, detect_nid, params);

    return detect_nid;
}

NodeID GraphBuilder::add_dummy_node(Graph &g, NodeParams params, NodeIdxPair input, TensorShape shape)
{
    return create_simple_single_input_output_node<DummyNode>(g, params, input, shape);
}

NodeID GraphBuilder::add_elementwise_node(Graph &g, NodeParams params, NodeIdxPair input0, NodeIdxPair input1, EltwiseOperation operation)
{
    check_nodeidx_pair(input0, g);
    check_nodeidx_pair(input1, g);

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
                                               NodeID weights_nid, NodeID bias_nid,
                                               const FullyConnectedLayerInfo fc_info, const QuantizationInfo out_quant_info)
{
    check_nodeidx_pair(input, g);
    ARM_COMPUTE_ERROR_ON(num_outputs == 0);
    ARM_COMPUTE_ERROR_ON(weights_nid == EmptyNodeID);

    const bool has_bias = (bias_nid != EmptyNodeID);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Create fully connected node and connect
    NodeID fc_nid = g.add_node<FullyConnectedLayerNode>(num_outputs, out_quant_info, fc_info);
    g.add_connection(input.node_id, input.index, fc_nid, 0);
    g.add_connection(weights_nid, 0, fc_nid, 1);
    if(has_bias)
    {
        g.add_connection(bias_nid, 0, fc_nid, 2);
    }

    set_node_params(g, fc_nid, params);

    return fc_nid;
}

NodeID GraphBuilder::add_fully_connected_layer(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_outputs,
                                               ITensorAccessorUPtr weights_accessor, ITensorAccessorUPtr bias_accessor,
                                               const FullyConnectedLayerInfo fc_info,
                                               const QuantizationInfo weights_quant_info, const QuantizationInfo out_quant_info)
{
    check_nodeidx_pair(input, g);
    ARM_COMPUTE_ERROR_ON(num_outputs == 0);

    bool has_bias = (bias_accessor != nullptr);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Create weights node
    TensorDescriptor w_desc = FullyConnectedLayerNode::compute_weights_descriptor(input_tensor_desc, num_outputs, fc_info, weights_quant_info);
    NodeID           w_nid  = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

    // Create bias nodes
    NodeID b_nid = EmptyNodeID;
    if(has_bias)
    {
        TensorDescriptor b_desc = input_tensor_desc;
        b_desc.shape            = TensorShape(num_outputs);
        if(is_data_type_quantized_asymmetric(input_tensor_desc.data_type))
        {
            b_desc.data_type = DataType::S32;
        }
        b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
    }

    // Create fully connected node and connect
    NodeID fc_nid = g.add_node<FullyConnectedLayerNode>(num_outputs, out_quant_info, fc_info);
    g.add_connection(input.node_id, input.index, fc_nid, 0);
    g.add_connection(w_nid, 0, fc_nid, 1);
    if(has_bias)
    {
        g.add_connection(b_nid, 0, fc_nid, 2);
    }

    set_node_params(g, fc_nid, params);

    return fc_nid;
}

NodeID GraphBuilder::add_generate_proposals_node(Graph &g, NodeParams params, NodeIdxPair scores, NodeIdxPair deltas, NodeIdxPair anchors, GenerateProposalsInfo info)
{
    check_nodeidx_pair(scores, g);
    check_nodeidx_pair(deltas, g);
    check_nodeidx_pair(anchors, g);

    NodeID nid = g.add_node<GenerateProposalsLayerNode>(info);

    g.add_connection(scores.node_id, scores.index, nid, 0);
    g.add_connection(deltas.node_id, deltas.index, nid, 1);
    g.add_connection(anchors.node_id, anchors.index, nid, 2);

    set_node_params(g, nid, params);
    return nid;
}

NodeID GraphBuilder::add_normalization_node(Graph &g, NodeParams params, NodeIdxPair input, NormalizationLayerInfo norm_info)
{
    return create_simple_single_input_output_node<NormalizationLayerNode>(g, params, input, norm_info);
}

NodeID GraphBuilder::add_normalize_planar_yuv_node(Graph &g, NodeParams params, NodeIdxPair input,
                                                   ITensorAccessorUPtr mean_accessor, ITensorAccessorUPtr std_accessor)
{
    check_nodeidx_pair(input, g);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

    // Calculate Common Descriptor
    TensorDescriptor common_desc = input_tensor_desc;
    common_desc.shape            = TensorShape(get_dimension_size(input_tensor_desc, DataLayoutDimension::CHANNEL));

    // Create mean and std nodes
    auto mean_nid = add_const_node_with_name(g, params, "Mean", common_desc, std::move(mean_accessor));
    auto std_nid  = add_const_node_with_name(g, params, "Std", common_desc, std::move(std_accessor));

    // Create normalize planar YUV node and add connections
    NodeID norm_planar_yuv_nid = g.add_node<NormalizePlanarYUVLayerNode>();
    g.add_connection(input.node_id, input.index, norm_planar_yuv_nid, 0);
    g.add_connection(mean_nid, 0, norm_planar_yuv_nid, 1);
    g.add_connection(std_nid, 0, norm_planar_yuv_nid, 2);
    set_node_params(g, norm_planar_yuv_nid, params);

    return norm_planar_yuv_nid;
}

NodeID GraphBuilder::add_pad_node(Graph &g, NodeParams params, NodeIdxPair input, PaddingList padding)
{
    return create_simple_single_input_output_node<PadLayerNode>(g, params, input, padding);
}

NodeID GraphBuilder::add_permute_node(Graph &g, NodeParams params, NodeIdxPair input, PermutationVector perm, DataLayout layout)
{
    return create_simple_single_input_output_node<PermuteLayerNode>(g, params, input, perm, layout);
}

NodeID GraphBuilder::add_pooling_node(Graph &g, NodeParams params, NodeIdxPair input, PoolingLayerInfo pool_info)
{
    return create_simple_single_input_output_node<PoolingLayerNode>(g, params, input, pool_info);
}

NodeID GraphBuilder::add_priorbox_node(Graph &g, NodeParams params, NodeIdxPair input0, NodeIdxPair input1, const PriorBoxLayerInfo &prior_info)
{
    check_nodeidx_pair(input0, g);
    check_nodeidx_pair(input1, g);

    // Create priorbox node and connect
    NodeID prior_nid = g.add_node<PriorBoxLayerNode>(prior_info);
    g.add_connection(input0.node_id, input0.index, prior_nid, 0);
    g.add_connection(input1.node_id, input1.index, prior_nid, 1);

    set_node_params(g, prior_nid, params);

    return prior_nid;
}

NodeID GraphBuilder::add_reorg_node(Graph &g, NodeParams params, NodeIdxPair input, int stride)
{
    return create_simple_single_input_output_node<ReorgLayerNode>(g, params, input, stride);
}

NodeID GraphBuilder::add_reshape_node(Graph &g, NodeParams params, NodeIdxPair input, TensorShape shape)
{
    return create_simple_single_input_output_node<ReshapeLayerNode>(g, params, input, shape);
}

NodeID GraphBuilder::add_resize_node(Graph &g, NodeParams params, NodeIdxPair input, InterpolationPolicy policy,
                                     float width_scale, float height_scale)
{
    return create_simple_single_input_output_node<ResizeLayerNode>(g, params, input, policy, width_scale, height_scale);
}

NodeID GraphBuilder::add_roi_align_node(Graph &g, NodeParams params, NodeIdxPair input, NodeIdxPair rois, ROIPoolingLayerInfo pool_info)
{
    check_nodeidx_pair(input, g);
    check_nodeidx_pair(rois, g);

    NodeID nid = g.add_node<ROIAlignLayerNode>(pool_info);

    g.add_connection(input.node_id, input.index, nid, 0);
    g.add_connection(rois.node_id, rois.index, nid, 1);

    set_node_params(g, nid, params);
    return nid;
}

NodeID GraphBuilder::add_scale_layer(Graph &g, const NodeParams &params, NodeIdxPair input, ITensorAccessorUPtr mul_accessor, ITensorAccessorUPtr add_accessor)
{
    check_nodeidx_pair(input, g);

    // Get input tensor descriptor
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
    const DataLayout       input_data_layout = input_tensor_desc.layout;

    // Create mul node
    TensorDescriptor mul_desc = input_tensor_desc;
    const size_t     C        = input_tensor_desc.shape[get_dimension_idx(input_data_layout, DataLayoutDimension::CHANNEL)];
    mul_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::WIDTH), 1);
    mul_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::HEIGHT), 1);
    mul_desc.shape.set(get_dimension_idx(input_data_layout, DataLayoutDimension::CHANNEL), C);
    NodeID      mul_const_nid   = add_const_node_with_name(g, params, "Mul", mul_desc, std::move(mul_accessor));
    NodeIdxPair mul_const_nidxp = { mul_const_nid, 0 };

    // Create add node
    TensorDescriptor add_desc        = mul_desc;
    NodeID           add_const_nid   = add_const_node_with_name(g, params, "Add", add_desc, std::move(add_accessor));
    NodeIdxPair      add_const_nidxp = { add_const_nid, 0 };

    // Create node and connect
    NodeID      mul_node      = GraphBuilder::add_elementwise_node(g, params, input, mul_const_nidxp, EltwiseOperation::Mul);
    NodeIdxPair mulnode_nidxp = { mul_node, 0 };
    NodeID      add_node      = GraphBuilder::add_elementwise_node(g, params, mulnode_nidxp, add_const_nidxp, EltwiseOperation::Add);

    return add_node;
}

NodeID GraphBuilder::add_softmax_node(Graph &g, NodeParams params, NodeIdxPair input, float beta)
{
    return create_simple_single_input_output_node<SoftmaxLayerNode>(g, params, input, beta);
}

NodeID GraphBuilder::add_slice_node(Graph &g, NodeParams params, NodeIdxPair input, Coordinates &starts, Coordinates &ends)
{
    return create_simple_single_input_output_node<SliceLayerNode>(g, params, input, starts, ends);
}

NodeID GraphBuilder::add_split_node(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_splits, unsigned int axis)
{
    return create_simple_single_input_output_node<SplitLayerNode>(g, params, input, num_splits, axis);
}

NodeID GraphBuilder::add_stack_node(Graph &g, NodeParams params, const std::vector<NodeIdxPair> &inputs, int axis)
{
    return create_simple_multiple_input_single_output_node<StackLayerNode>(g, params, inputs, inputs.size(), axis);
}

NodeID GraphBuilder::add_upsample_node(Graph &g, NodeParams params, NodeIdxPair input, Size2D info, InterpolationPolicy upsampling_policy)
{
    return create_simple_single_input_output_node<UpsampleLayerNode>(g, params, input, info, upsampling_policy);
}

NodeID GraphBuilder::add_yolo_node(Graph &g, NodeParams params, NodeIdxPair input, ActivationLayerInfo act_info, int32_t num_classes)
{
    return create_simple_single_input_output_node<YOLOLayerNode>(g, params, input, act_info, num_classes);
}
} // namespace graph
} // namespace arm_compute
