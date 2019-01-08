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
#ifndef __ARM_COMPUTE_GRAPH_GRAPH_BUILDER_H__
#define __ARM_COMPUTE_GRAPH_GRAPH_BUILDER_H__

#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/LayerDescriptors.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
// Forward declaration
class Graph;

/** Graph builder class
 *
 * Builds and compiles a graph
 */
class GraphBuilder final
{
public:
    /** Adds a Const node to the graph
     *
     * @param[in] g        Graph to add the node to
     * @param[in] params   Common node parameters
     * @param[in] desc     Tensor descriptor of the node
     * @param[in] accessor (Optional) Accessor of the const node data
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_const_node(Graph &g, NodeParams params, const TensorDescriptor &desc, ITensorAccessorUPtr accessor = nullptr);
    /** Adds an input layer node to the graph
     *
     * @param[in] g        Graph to add the node to
     * @param[in] params   Common node parameters
     * @param[in] desc     Tensor descriptor of the Tensor
     * @param[in] accessor (Optional) Accessor of the input node data
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_input_node(Graph &g, NodeParams params, const TensorDescriptor &desc, ITensorAccessorUPtr accessor = nullptr);
    /** Adds an output layer node to the graph
     *
     * @param[in] g        Graph to add the node to
     * @param[in] params   Common node parameters
     * @param[in] input    Input to the output node as a NodeID-Index pair
     * @param[in] accessor (Optional) Accessor of the output node data
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_output_node(Graph &g, NodeParams params, NodeIdxPair input, ITensorAccessorUPtr accessor = nullptr);
    /** Adds an activation layer node to the graph
     *
     * @param[in] g              Graph to add the node to
     * @param[in] params         Common node parameters
     * @param[in] input          Input to the activation layer node as a NodeID-Index pair
     * @param[in] act_info       Activation layer information
     * @param[in] out_quant_info (Optional) Output quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_activation_node(Graph &g, NodeParams params, NodeIdxPair input, ActivationLayerInfo act_info,
                                      const QuantizationInfo &out_quant_info = QuantizationInfo());
    /** Adds a batch normalization layer node to the graph
     *
     * @param[in] g              Graph to add the node to
     * @param[in] params         Common node parameters
     * @param[in] input          Input to the batch normalization layer node as a NodeID-Index pair
     * @param[in] epsilon        Epsilon parameter
     * @param[in] mean_accessor  Const Node ID that contains the mean values
     * @param[in] var_accessor   Const Node ID that contains the variance values
     * @param[in] beta_accessor  Const Node ID that contains the beta values. Can be EmptyNodeID
     * @param[in] gamma_accessor Const Node ID that contains the gamma values. Can be EmptyNodeID
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_batch_normalization_node(Graph &g, NodeParams params, NodeIdxPair input, float epsilon,
                                               ITensorAccessorUPtr mean_accessor = nullptr, ITensorAccessorUPtr var_accessor = nullptr,
                                               ITensorAccessorUPtr beta_accessor = nullptr, ITensorAccessorUPtr gamma_accessor = nullptr);
    /** Adds a bounding box transform layer node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the bounding box transform layer node as a NodeID-Index pair
     * @param[in] deltas Deltas input to the bounding box transform layer node as a NodeID-Index pair
     * @param[in] info   Bounding Box Transform information
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_bounding_box_transform_node(Graph &g, NodeParams params, NodeIdxPair input, NodeIdxPair deltas, BoundingBoxTransformInfo info);
    /** Adds an channel shuffle layer node to the graph
     *
     * @param[in] g          Graph to add the node to
     * @param[in] params     Common node parameters
     * @param[in] input      Input to the activation layer node as a NodeID-Index pair
     * @param[in] num_groups Number of groups
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_channel_shuffle_node(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_groups);
    /** Adds a convolution layer node to the graph
     *
     * TODO (COMPMID-1113): Add a graph descriptor for convolution layer node
     *
     * @param[in] g                     Graph to add the node to
     * @param[in] params                Common node parameters
     * @param[in] input                 Input to the convolution layer node as a NodeID-Index pair
     * @param[in] kernel_spatial_extend Spatial extend of convolution kernels
     * @param[in] depth                 Number of convolution kernels
     * @param[in] conv_info             Convolution layer information
     * @param[in] num_groups            (Optional) Number of groups for a grouped convolution. Defaults to 1
     * @param[in] method                (Optional) Convolution method to use
     * @param[in] fast_math_hint        (Optional) Fast math hint
     * @param[in] weights_accessor      (Optional) Accessor of the weights node data
     * @param[in] bias_accessor         (Optional) Accessor of the bias node data
     * @param[in] weights_quant_info    (Optional) Weights quantization info
     * @param[in] out_quant_info        (Optional) Output quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_convolution_node(Graph &g, NodeParams params, NodeIdxPair input,
                                       Size2D kernel_spatial_extend, unsigned int depth, PadStrideInfo conv_info, unsigned int num_groups = 1,
                                       ConvolutionMethod method = ConvolutionMethod::Default, FastMathHint fast_math_hint = FastMathHint::Disabled,
                                       ITensorAccessorUPtr weights_accessor = nullptr, ITensorAccessorUPtr bias_accessor = nullptr,
                                       const QuantizationInfo &weights_quant_info = QuantizationInfo(),
                                       const QuantizationInfo &out_quant_info     = QuantizationInfo());
    /** Adds a deconvolution layer node to the graph
     *
     * @param[in] g                     Graph to add the node to
     * @param[in] params                Common node parameters
     * @param[in] input                 Input to the convolution layer node as a NodeID-Index pair
     * @param[in] kernel_spatial_extend Spatial extend of convolution kernels
     * @param[in] depth                 Number of convolution kernels
     * @param[in] deconv_info           Convolution layer information
     * @param[in] weights_accessor      (Optional) Accessor of the weights node data
     * @param[in] bias_accessor         (Optional) Accessor of the bias node data
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_deconvolution_node(Graph &g, NodeParams params, NodeIdxPair input,
                                         Size2D kernel_spatial_extend, unsigned int depth, PadStrideInfo deconv_info,
                                         ITensorAccessorUPtr weights_accessor = nullptr, ITensorAccessorUPtr bias_accessor = nullptr);
    /** Adds a depth concatenate node to the graph
     *
     * @param[in] g                 Graph to add the node to
     * @param[in] params            Common node parameters
     * @param[in] inputs            Inputs to the concatenate layer node as a NodeID-Index pair
     * @param[in] concat_descriptor Concatenation layer descriptor
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_concatenate_node(Graph &g, NodeParams params, const std::vector<NodeIdxPair> &inputs, const descriptors::ConcatLayerDescriptor &concat_descriptor);
    /** Adds a depth-wise convolution layer node to the graph
     *
     * @param[in] g                     Graph to add the node to
     * @param[in] params                Common node parameters
     * @param[in] input                 Input to the depthwise convolution layer node as a NodeID-Index pair
     * @param[in] kernel_spatial_extend Spatial extend of convolution kernels
     * @param[in] conv_info             Convolution layer information
     * @param[in] depth_multiplier      (Optional) Depth multiplier parameter.
     * @param[in] method                (Optional) Convolution method to use
     * @param[in] weights_accessor      (Optional) Accessor of the weights node data
     * @param[in] bias_accessor         (Optional) Accessor of the bias node data
     * @param[in] quant_info            (Optional) Weights quantization info
     * @param[in] out_quant_info        (Optional) Output quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_depthwise_convolution_node(Graph &g, NodeParams params, NodeIdxPair input,
                                                 Size2D kernel_spatial_extend, PadStrideInfo conv_info, int depth_multiplier = 1,
                                                 DepthwiseConvolutionMethod method    = DepthwiseConvolutionMethod::Default,
                                                 ITensorAccessorUPtr weights_accessor = nullptr, ITensorAccessorUPtr bias_accessor = nullptr, const QuantizationInfo &quant_info = QuantizationInfo(),
                                                 const QuantizationInfo &out_quant_info = QuantizationInfo());
    /** Adds an element-wise layer node to the graph
     *
     * @param[in] g         Graph to add the node to
     * @param[in] params    Common node parameters
     * @param[in] input0    First input to the element-wise operation layer node as a NodeID-Index pair
     * @param[in] input1    Second input to the element-wise operation layer node as a NodeID-Index pair
     * @param[in] operation Element-wise operation to perform
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_elementwise_node(Graph &g, NodeParams params, NodeIdxPair input0, NodeIdxPair input1, EltwiseOperation operation);
    /** Adds a detection output layer node to the graph
     *
     * @param[in] g              Graph to add the node to
     * @param[in] params         Common node parameters
     * @param[in] input_loc      Location input to the detection output layer node as a NodeID-Index pair
     * @param[in] input_conf     Confidence input to the detection output layer node as a NodeID-Index pair
     * @param[in] input_priorbox PriorBox input to the detection output layer node as a NodeID-Index pair
     * @param[in] detect_info    Detection output layer parameters
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_detection_output_node(Graph &g, NodeParams params, NodeIdxPair input_loc, NodeIdxPair input_conf, NodeIdxPair input_priorbox, const DetectionOutputLayerInfo &detect_info);
    /** Adds a detection post process layer node to the graph
     *
     * @param[in] g                      Graph to add the node to
     * @param[in] params                 Common node parameters
     * @param[in] input_box_encoding     Boxes input to the detection output layer node as a NodeID-Index pair
     * @param[in] input_class_prediction Class prediction input to the detection output layer node as a NodeID-Index pair
     * @param[in] detect_info            Detection output layer parameters
     * @param[in] anchors_accessor       (Optional) Const Node ID that contains the anchor values
     * @param[in] anchor_quant_info      (Optional) Anchor quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_detection_post_process_node(Graph &g, NodeParams params, NodeIdxPair input_box_encoding, NodeIdxPair input_class_prediction,
                                                  const DetectionPostProcessLayerInfo &detect_info, ITensorAccessorUPtr anchors_accessor = nullptr,
                                                  const QuantizationInfo &anchor_quant_info = QuantizationInfo());
    /** Adds a Dummy node to the graph
     *
     * @note this node if for debugging purposes. Just alters the shape of the graph pipeline as requested.
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the dummy node as a NodeID-Index pair
     * @param[in] shape  Output shape
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_dummy_node(Graph &g, NodeParams params, NodeIdxPair input, TensorShape shape);
    /** Adds a flatten layer node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the flatten layer node as a NodeID-Index pair
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_flatten_node(Graph &g, NodeParams params, NodeIdxPair input);
    /** Adds a fully connected layer node to the graph
     *
     * @param[in] g              Graph to add the layer to
     * @param[in] params         Common node parameters
     * @param[in] input          Input to the fully connected layer node as a NodeID-Index pair
     * @param[in] num_outputs    Number of output neurons
     * @param[in] weights_nid    Node ID of the weights node data
     * @param[in] bias_nid       (Optional) Node ID of the bias node data. Defaults to EmptyNodeID
     * @param[in] fc_info        (Optional) Fully connected layer metadata
     * @param[in] out_quant_info (Optional) Output quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_fully_connected_layer(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_outputs,
                                            NodeID weights_nid, NodeID bias_nid = EmptyNodeID,
                                            const FullyConnectedLayerInfo fc_info        = FullyConnectedLayerInfo(),
                                            const QuantizationInfo       &out_quant_info = QuantizationInfo());
    /** Adds a fully connected layer node to the graph
     *
     * @param[in] g                  Graph to add the layer to
     * @param[in] params             Common node parameters
     * @param[in] input              Input to the fully connected layer node as a NodeID-Index pair
     * @param[in] num_outputs        Number of output neurons
     * @param[in] weights_accessor   (Optional) Accessor of the weights node data
     * @param[in] bias_accessor      (Optional) Accessor of the bias node data
     * @param[in] fc_info            (Optional) Fully connected layer metadata
     * @param[in] weights_quant_info (Optional) Weights quantization info
     * @param[in] out_quant_info     (Optional) Output quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_fully_connected_layer(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_outputs,
                                            ITensorAccessorUPtr weights_accessor = nullptr, ITensorAccessorUPtr bias_accessor = nullptr,
                                            const FullyConnectedLayerInfo fc_info            = FullyConnectedLayerInfo(),
                                            const QuantizationInfo       &weights_quant_info = QuantizationInfo(),
                                            const QuantizationInfo       &out_quant_info     = QuantizationInfo());
    /** Adds a generate proposals layer node to the graph
     *
     * @param[in] g       Graph to add the layer to
     * @param[in] params  Common node parameters
     * @param[in] scores  Input scores to the generate proposals layer node as a NodeID-Index pair
     * @param[in] deltas  Input deltas to the generate proposals layer node as a NodeID-Index pair
     * @param[in] anchors Input anchors to the generate proposals layer node as a NodeID-Index pair
     * @param[in] info    Generate proposals operation information
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_generate_proposals_node(Graph &g, NodeParams params, NodeIdxPair scores, NodeIdxPair deltas,
                                              NodeIdxPair anchors, GenerateProposalsInfo info);
    /** Adds a normalization layer node to the graph
     *
     * @param[in] g         Graph to add the node to
     * @param[in] params    Common node parameters
     * @param[in] input     Input to the normalization layer node as a NodeID-Index pair
     * @param[in] norm_info Normalization layer information
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_normalization_node(Graph &g, NodeParams params, NodeIdxPair input, NormalizationLayerInfo norm_info);
    /** Adds a normalize planar YUV layer node to the graph
     *
     * @param[in] g             Graph to add the node to
     * @param[in] params        Common node parameters
     * @param[in] input         Input to the normalize planar YUV layer node as a NodeID-Index pair
     * @param[in] mean_accessor Const Node ID that contains the mean values
     * @param[in] std_accessor  Const Node ID that contains the variance values
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_normalize_planar_yuv_node(Graph &g, NodeParams params, NodeIdxPair input,
                                                ITensorAccessorUPtr mean_accessor = nullptr, ITensorAccessorUPtr std_accessor = nullptr);
    /** Adds a pad layer node to the graph
     *
     * @param[in] g       Graph to add the node to
     * @param[in] params  Common node parameters
     * @param[in] input   Input to the reshape layer node as a NodeID-Index pair
     * @param[in] padding The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                    specifies the front and the end padding in the i-th dimension.
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_pad_node(Graph &g, NodeParams params, NodeIdxPair input, PaddingList padding);
    /** Adds a permute layer node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the reshape layer node as a NodeID-Index pair
     * @param[in] perm   Permutation vector
     * @param[in] layout (Optional) Data layout to assign to permuted tensor.
     *                    If UNKNOWN then the input's layout will be used.
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_permute_node(Graph &g, NodeParams params, NodeIdxPair input, PermutationVector perm, DataLayout layout = DataLayout::UNKNOWN);
    /** Adds a pooling layer node to the graph
     *
     * @param[in] g         Graph to add the node to
     * @param[in] params    Common node parameters
     * @param[in] input     Input to the pooling layer node as a NodeID-Index pair
     * @param[in] pool_info Pooling layer information
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_pooling_node(Graph &g, NodeParams params, NodeIdxPair input, PoolingLayerInfo pool_info);
    /** Adds a priorbox layer node to the graph
     *
     * @param[in] g          Graph to add the node to
     * @param[in] params     Common node parameters
     * @param[in] input0     First input to the priorbox layer node as a NodeID-Index pair
     * @param[in] input1     Second input to the priorbox layer node as a NodeID-Index pair
     * @param[in] prior_info PriorBox parameters
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_priorbox_node(Graph &g, NodeParams params, NodeIdxPair input0, NodeIdxPair input1, const PriorBoxLayerInfo &prior_info);
    /** Adds a quantization layer node to the graph
     *
     * @param[in] g              Graph to add the node to
     * @param[in] params         Common node parameters
     * @param[in] input          Input to the quantization layer node as a NodeID-Index pair
     * @param[in] out_quant_info Output quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_quantization_node(Graph &g, NodeParams params, NodeIdxPair input, const QuantizationInfo &out_quant_info);
    /** Adds a reorg layer node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the reorg layer node as a NodeID-Index pair
     * @param[in] stride Stride value to use for reorganizing the values in the output tensor.
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_reorg_node(Graph &g, NodeParams params, NodeIdxPair input, int stride);
    /** Adds a reshape layer node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the reshape layer node as a NodeID-Index pair
     * @param[in] shape  Output reshaped shape
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_reshape_node(Graph &g, NodeParams params, NodeIdxPair input, TensorShape shape);
    /** Adds a resize layer node to the graph
     *
     * @param[in] g            Graph to add the node to
     * @param[in] params       Common node parameters
     * @param[in] input        Input to the reshape layer node as a NodeID-Index pair
     * @param[in] policy       Interpolation policy
     * @param[in] width_scale  Width scaling factor
     * @param[in] height_scale Height scaling factor
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_resize_node(Graph &g, NodeParams params, NodeIdxPair input, InterpolationPolicy policy, float width_scale, float height_scale);
    /** Adds a ROI align layer node to the graph
     *
     * @param[in] g         Graph to add the node to
     * @param[in] params    Common node parameters
     * @param[in] input     Input to the reshape layer node as a NodeID-Index pair
     * @param[in] rois      Input containing the ROIs.
     * @param[in] pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_roi_align_node(Graph &g, NodeParams params, NodeIdxPair input, NodeIdxPair rois, ROIPoolingLayerInfo pool_info);
    /** Adds a scale layer node to the graph
     * This layer computes a product of the input with a scale (read from mul_accessor) and it applies an offset (read from add_accessor).
     * output = input * mul_w + add_w
     *
     * @param[in] g            Graph to add the layer to
     * @param[in] params       Common node parameters
     * @param[in] input        Input to the fully connected layer node as a NodeID-Index pair
     * @param[in] mul_accessor (Optional) Accessor of the mul node data
     * @param[in] add_accessor (Optional) Accessor of the add node data
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_scale_layer(Graph &g, const NodeParams &params, NodeIdxPair input,
                                  ITensorAccessorUPtr mul_accessor = nullptr, ITensorAccessorUPtr add_accessor = nullptr);
    /** Adds a softmax node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the softmax layer node as a NodeID-Index pair
     * @param[in] beta   Beta parameter
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_softmax_node(Graph &g, NodeParams params, NodeIdxPair input, float beta = 1.f);
    /** Adds a slice node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] input  Input to the slice layer node as a NodeID-Index pair
     * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_slice_node(Graph &g, NodeParams params, NodeIdxPair input, Coordinates &starts, Coordinates &ends);
    /** Adds a split node to the graph
     *
     * @param[in] g          Graph to add the node to
     * @param[in] params     Common node parameters
     * @param[in] input      Input to the split layer node as a NodeID-Index pair
     * @param[in] num_splits Number of different splits
     * @param[in] axis       (Optional) Split axis. Defaults to 0
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_split_node(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_splits, unsigned int axis = 0);
    /** Adds a stack layer node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] inputs Inputs to the reorg layer node as a NodeID-Index pair
     * @param[in] axis   Axis along which the input tensors have to be packed
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_stack_node(Graph &g, NodeParams params, const std::vector<NodeIdxPair> &inputs, int axis);
    /** Adds an upsample layer to the graph
     *
     * @param[in] g                 Graph to add the node to
     * @param[in] params            Common node parameters
     * @param[in] input             Input to the yolo layer node as a NodeID-Index pair
     * @param[in] info              Upsample layer stride info
     * @param[in] upsampling_policy Upsampling policy used
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_upsample_node(Graph &g, NodeParams params, NodeIdxPair input, Size2D info, InterpolationPolicy upsampling_policy);
    /** Adds a yolo layer to the graph
     *
     * @param[in] g           Graph to add the node to
     * @param[in] params      Common node parameters
     * @param[in] input       Input to the yolo layer node as a NodeID-Index pair
     * @param[in] act_info    Activation layer parameters
     * @param[in] num_classes Number of classes to activate
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_yolo_node(Graph &g, NodeParams params, NodeIdxPair input, ActivationLayerInfo act_info, int32_t num_classes);
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_GRAPH_BUILDER_H__ */
