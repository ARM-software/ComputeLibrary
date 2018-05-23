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
#ifndef __ARM_COMPUTE_GRAPH_GRAPH_BUILDER_H__
#define __ARM_COMPUTE_GRAPH_GRAPH_BUILDER_H__

#include "arm_compute/graph/ITensorAccessor.h"
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
    static NodeID add_const_node(Graph &g, NodeParams params, TensorDescriptor desc, ITensorAccessorUPtr accessor = nullptr);
    /** Adds an input layer node to the graph
     *
     * @param[in] g        Graph to add the node to
     * @param[in] params   Common node parameters
     * @param[in] desc     Tensor descriptor of the Tensor
     * @param[in] accessor (Optional) Accessor of the input node data
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_input_node(Graph &g, NodeParams params, TensorDescriptor desc, ITensorAccessorUPtr accessor = nullptr);
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
     * @param[in] g        Graph to add the node to
     * @param[in] params   Common node parameters
     * @param[in] input    Input to the activation layer node as a NodeID-Index pair
     * @param[in] act_info Activation layer information
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_activation_node(Graph &g, NodeParams params, NodeIdxPair input, ActivationLayerInfo act_info);
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
    /** Adds a convolution layer node to the graph
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
                                       Size2D kernel_spatial_extend, unsigned int depth, PadStrideInfo conv_info,
                                       unsigned int num_groups = 1, ConvolutionMethod method = ConvolutionMethod::DEFAULT, FastMathHint fast_math_hint = FastMathHint::DISABLED,
                                       ITensorAccessorUPtr weights_accessor = nullptr, ITensorAccessorUPtr bias_accessor = nullptr,
                                       const QuantizationInfo weights_quant_info = QuantizationInfo(),
                                       const QuantizationInfo out_quant_info     = QuantizationInfo());
    /** Adds a depth concatenate node to the graph
     *
     * @param[in] g      Graph to add the node to
     * @param[in] params Common node parameters
     * @param[in] inputs Inputs to the depth concatenate layer node as a NodeID-Index pair
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_depth_concatenate_node(Graph &g, NodeParams params, std::vector<NodeIdxPair> inputs);
    /** Adds a depth-wise convolution layer node to the graph
     *
     * @param[in] g                     Graph to add the node to
     * @param[in] params                Common node parameters
     * @param[in] input                 Input to the depthwise convolution layer node as a NodeID-Index pair
     * @param[in] kernel_spatial_extend Spatial extend of convolution kernels
     * @param[in] conv_info             Convolution layer information
     * @param[in] method                (Optional) Convolution method to use
     * @param[in] weights_accessor      (Optional) Accessor of the weights node data
     * @param[in] bias_accessor         (Optional) Accessor of the bias node data
     * @param[in] quant_info            (Optional) Weights quantization info
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_depthwise_convolution_node(Graph &g, NodeParams params, NodeIdxPair input,
                                                 Size2D kernel_spatial_extend, PadStrideInfo conv_info,
                                                 DepthwiseConvolutionMethod method    = DepthwiseConvolutionMethod::DEFAULT,
                                                 ITensorAccessorUPtr weights_accessor = nullptr, ITensorAccessorUPtr bias_accessor = nullptr, const QuantizationInfo quant_info = QuantizationInfo());
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
     * @param[in] g                Graph to add the layer to
     * @param[in] params           Common node parameters
     * @param[in] input            Input to the fully connected layer node as a NodeID-Index pair
     * @param[in] num_outputs      Number of output neurons
     * @param[in] weights_accessor (Optional) Accessor of the weights node data
     * @param[in] bias_accessor    (Optional) Accessor of the bias node data
     *
     * @return Node ID of the created node, EmptyNodeID in case of error
     */
    static NodeID add_fully_connected_layer(Graph &g, NodeParams params, NodeIdxPair input, unsigned int num_outputs,
                                            ITensorAccessorUPtr weights_accessor = nullptr, ITensorAccessorUPtr bias_accessor = nullptr);
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
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_GRAPH_BUILDER_H__ */
