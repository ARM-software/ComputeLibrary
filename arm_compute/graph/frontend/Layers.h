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
#ifndef ARM_COMPUTE_GRAPH_LAYERS_H
#define ARM_COMPUTE_GRAPH_LAYERS_H

#include "arm_compute/graph/GraphBuilder.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/graph/frontend/ILayer.h"
#include "arm_compute/graph/frontend/IStream.h"
#include "arm_compute/graph/frontend/SubStream.h"

#include "arm_compute/core/utils/misc/Utility.h"

#include <memory>
#include <string>

namespace arm_compute
{
namespace graph
{
namespace frontend
{
/** Input Layer */
class InputLayer final : public ILayer
{
public:
    /** Construct an input layer.
     *
     * @param[in] desc     Description of input tensor.
     * @param[in] accessor Accessor to get input tensor data from.
     */
    InputLayer(TensorDescriptor desc, ITensorAccessorUPtr accessor)
        : _desc(desc), _accessor(std::move(accessor))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams common_params = { name(), s.hints().target_hint };
        return GraphBuilder::add_input_node(s.graph(), common_params, _desc, std::move(_accessor));
    }

private:
    TensorDescriptor    _desc;
    ITensorAccessorUPtr _accessor;
};

/** Constant Layer */
class ConstantLayer final : public ILayer
{
public:
    /** Construct a constant layer.
     *
     * @param[in] desc     Description of input tensor.
     * @param[in] accessor Accessor to get input tensor data from.
     */
    ConstantLayer(TensorDescriptor desc, ITensorAccessorUPtr accessor)
        : _desc(desc), _accessor(std::move(accessor))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams common_params = { name(), s.hints().target_hint };
        return GraphBuilder::add_const_node(s.graph(), common_params, _desc, std::move(_accessor));
    }

private:
    TensorDescriptor    _desc;
    ITensorAccessorUPtr _accessor;
};

/** Output Layer */
class OutputLayer final : public ILayer
{
public:
    /** Construct an output layer.
     *
     * @param[in] accessor       Accessor to give output tensor data to.
     * @param[in] connection_idx (Optional) Input connection index
     */
    OutputLayer(ITensorAccessorUPtr accessor, unsigned int connection_idx = 0)
        : _accessor(std::move(accessor)), _connection_idx(connection_idx)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), _connection_idx };
        return GraphBuilder::add_output_node(s.graph(), common_params, input, std::move(_accessor));
    }

private:
    ITensorAccessorUPtr _accessor;
    unsigned int        _connection_idx;
};

/** Activation Layer */
class ActivationLayer final : public ILayer
{
public:
    /** Construct an activation layer.
     *
     * @param[in] act_info       Activation information
     * @param[in] out_quant_info (Optional) Output quantization info
     */
    ActivationLayer(ActivationLayerInfo    act_info,
                    const QuantizationInfo out_quant_info = QuantizationInfo())
        : _act_info(act_info),
          _out_quant_info(std::move(out_quant_info))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_activation_node(s.graph(), common_params, input, _act_info, std::move(_out_quant_info));
    }

private:
    ActivationLayerInfo    _act_info;
    const QuantizationInfo _out_quant_info;
};

/** ArgMinMax Layer */
class ArgMinMaxLayer final : public ILayer
{
public:
    /** Construct an activation layer.
     *
     * @param[in] op             Reduction Operation: min or max
     * @param[in] axis           Axis to perform reduction along
     * @param[in] out_data_type  (Optional) Output tensor data type
     * @param[in] out_quant_info (Optional) Output quantization info
     */
    ArgMinMaxLayer(ReductionOperation     op,
                   unsigned int           axis,
                   DataType               out_data_type  = DataType::UNKNOWN,
                   const QuantizationInfo out_quant_info = QuantizationInfo())
        : _op(op),
          _axis(axis),
          _out_data_type(out_data_type),
          _out_quant_info(std::move(out_quant_info))
    {
    }

    /** Create layer and add to the given stream.
     *
     * @param[in] s Stream to add layer to.
     *
     * @return ID of the created node.
     */
    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_arg_min_max_node(s.graph(), common_params, input, _op, _axis, _out_data_type, std::move(_out_quant_info));
    }

private:
    ReductionOperation _op;
    unsigned int       _axis;
    DataType           _out_data_type;
    QuantizationInfo   _out_quant_info;
};

/** Batchnormalization Layer */
class BatchNormalizationLayer final : public ILayer
{
public:
    /** Construct a batch normalization layer.
     *
     * @param[in] mean    Accessor to get mean tensor data from.
     * @param[in] var     Accessor to get var tensor data from.
     * @param[in] gamma   (Optional) Accessor to get gamma tensor data from. Default: nullptr.
     * @param[in] beta    (Optional) Accessor to get beta tensor data from. Default: nullptr.
     * @param[in] epsilon (Optional) Epsilon value. Default: 0.001.
     */
    BatchNormalizationLayer(ITensorAccessorUPtr mean,
                            ITensorAccessorUPtr var,
                            ITensorAccessorUPtr gamma   = nullptr,
                            ITensorAccessorUPtr beta    = nullptr,
                            float               epsilon = 0.001f)
        : _mean(std::move(mean)), _var(std::move(var)), _gamma(std::move(gamma)), _beta(std::move(beta)), _epsilon(epsilon)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        ARM_COMPUTE_ERROR_ON(_mean == nullptr);
        ARM_COMPUTE_ERROR_ON(_var == nullptr);

        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_batch_normalization_node(s.graph(), common_params, input, _epsilon,
                                                          std::move(_mean), std::move(_var), std::move(_beta), std::move(_gamma));
    }

private:
    ITensorAccessorUPtr _mean;
    ITensorAccessorUPtr _var;
    ITensorAccessorUPtr _gamma;
    ITensorAccessorUPtr _beta;
    float               _epsilon;
};

/** Bounding Box Transform Layer */
class BoundingBoxTransformLayer final : public ILayer
{
public:
    /** Construct a bounding box transform layer.
     *
     * @param[in] sub_stream_input  Graph sub-stream for the input
     * @param[in] sub_stream_deltas Graph sub-stream for the deltas
     * @param[in] info              Contains BoundingBox operation information described in @ref BoundingBoxTransformInfo.
     */
    BoundingBoxTransformLayer(SubStream &&sub_stream_input, SubStream &&sub_stream_deltas, BoundingBoxTransformInfo info)
        : _ss_input(sub_stream_input), _ss_deltas(sub_stream_deltas), _bbox_info(info)
    {
    }

    /** Create layer and add to the given stream.
     *
     * @param[in] s Stream to add layer to.
     *
     * @return ID of the created node.
     */
    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { _ss_input.tail_node(), 0 };
        NodeIdxPair deltas        = { _ss_deltas.tail_node(), 0 };
        return GraphBuilder::add_bounding_box_transform_node(s.graph(), common_params, input, deltas, _bbox_info);
    }

private:
    SubStream                _ss_input;
    SubStream                _ss_deltas;
    BoundingBoxTransformInfo _bbox_info;
};

/** Channel Shuffle Layer */
class ChannelShuffleLayer final : public ILayer
{
public:
    /** Construct a Channel Shuffle layer.
     *
     * @param[in] num_groups Number of groups
     */
    ChannelShuffleLayer(unsigned int num_groups)
        : _num_groups(num_groups)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_channel_shuffle_node(s.graph(), common_params, input, _num_groups);
    }

private:
    unsigned int _num_groups;
};

/** Concat Layer */
class ConcatLayer final : public ILayer
{
public:
    /** Construct a concatenation layer
     *
     * @param[in] sub_stream1      First graph branch
     * @param[in] sub_stream2      Second graph branch
     * @param[in] rest_sub_streams Rest sub-graph branches
     */
    template <typename... Ts>
    ConcatLayer(SubStream &&sub_stream1, SubStream &&sub_stream2, Ts &&... rest_sub_streams)
        : _sub_streams(), _concat_descriptor(DataLayoutDimension::CHANNEL)
    {
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream1)));
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream2)));

        utility::for_each([&](SubStream && sub_stream)
        {
            _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream)));
        },
        std::move(rest_sub_streams)...);
    }
    /** Construct a concatenation layer
     *
     * @param[in] concat_descriptor Concat layer descriptor
     * @param[in] sub_stream1       First graph branch
     * @param[in] sub_stream2       Second graph branch
     * @param[in] rest_sub_streams  Rest sub-graph branches
     */
    template <typename... Ts>
    ConcatLayer(descriptors::ConcatLayerDescriptor concat_descriptor, SubStream &&sub_stream1, SubStream &&sub_stream2, Ts &&... rest_sub_streams)
        : _sub_streams(), _concat_descriptor(concat_descriptor)
    {
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream1)));
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream2)));

        utility::for_each([&](SubStream && sub_stream)
        {
            _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream)));
        },
        std::move(rest_sub_streams)...);
    }
    /** Construct a concat layer
     *
     * @param[in] sub_stream Sub-stream
     */
    template <typename... Ts>
    ConcatLayer(SubStream &&sub_stream)
        : _sub_streams(), _concat_descriptor(DataLayoutDimension::CHANNEL)
    {
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream)));
    }
    NodeID create_layer(IStream &s) override
    {
        NodeID     nid           = EmptyNodeID;
        NodeParams common_params = { name(), s.hints().target_hint };
        if(_sub_streams.size() == 1 && _sub_streams.at(0) != nullptr)
        {
            nid = _sub_streams[0]->tail_node();
        }
        else
        {
            // Collect tail nodes and concatenate
            std::vector<NodeIdxPair> nodes;
            for(auto &ss : _sub_streams)
            {
                if(ss && (ss->tail_node() != EmptyNodeID))
                {
                    const auto tail_node = s.graph().node(ss->tail_node());
                    if(tail_node != nullptr && tail_node->type() != NodeType::Output)
                    {
                        nodes.push_back({ ss->tail_node(), 0 });
                    }
                }
            }
            nid = GraphBuilder::add_concatenate_node(s.graph(), common_params, nodes, _concat_descriptor);
        }
        return nid;
    }

private:
    std::vector<std::unique_ptr<SubStream>> _sub_streams;
    descriptors::ConcatLayerDescriptor      _concat_descriptor;
};

/** Convolution Layer */
class ConvolutionLayer final : public ILayer
{
public:
    /** Construct a convolution layer.
     *
     * @param[in] conv_width         Convolution width.
     * @param[in] conv_height        Convolution height.
     * @param[in] ofm                Output feature map.
     * @param[in] weights            Accessor to get kernel weights from.
     * @param[in] bias               Accessor to get kernel bias from.
     * @param[in] conv_info          Padding and stride information.
     * @param[in] num_groups         (Optional) Number of groups. Default: 1.
     * @param[in] weights_quant_info (Optional) Weights quantization information
     * @param[in] out_quant_info     (Optional) Output quantization info
     */
    ConvolutionLayer(unsigned int           conv_width,
                     unsigned int           conv_height,
                     unsigned int           ofm,
                     ITensorAccessorUPtr    weights,
                     ITensorAccessorUPtr    bias,
                     PadStrideInfo          conv_info,
                     unsigned int           num_groups         = 1,
                     const QuantizationInfo weights_quant_info = QuantizationInfo(),
                     const QuantizationInfo out_quant_info     = QuantizationInfo())
        : _conv_width(conv_width),
          _conv_height(conv_height),
          _ofm(ofm),
          _conv_info(std::move(conv_info)),
          _num_groups(num_groups),
          _weights(std::move(weights)),
          _bias(std::move(bias)),
          _weights_quant_info(std::move(weights_quant_info)),
          _out_quant_info(std::move(out_quant_info))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeIdxPair input         = { s.tail_node(), 0 };
        NodeParams  common_params = { name(), s.hints().target_hint };
        return GraphBuilder::add_convolution_node(s.graph(), common_params, input,
                                                  Size2D(_conv_width, _conv_height), _ofm, _conv_info, _num_groups,
                                                  s.hints().convolution_method_hint, s.hints().fast_math_hint,
                                                  std::move(_weights), std::move(_bias), std::move(_weights_quant_info), std::move(_out_quant_info));
    }

private:
    unsigned int           _conv_width;
    unsigned int           _conv_height;
    unsigned int           _ofm;
    const PadStrideInfo    _conv_info;
    unsigned int           _num_groups;
    ITensorAccessorUPtr    _weights;
    ITensorAccessorUPtr    _bias;
    const QuantizationInfo _weights_quant_info;
    const QuantizationInfo _out_quant_info;
};

/** Deconvolution Layer */
class DeconvolutionLayer final : public ILayer
{
public:
    /** Construct a convolution layer.
     *
     * @param[in] conv_width  Convolution width.
     * @param[in] conv_height Convolution height.
     * @param[in] ofm         Output feature map.
     * @param[in] weights     Accessor to get kernel weights from.
     * @param[in] bias        Accessor to get kernel bias from.
     * @param[in] deconv_info Padding and stride information.
     */
    DeconvolutionLayer(unsigned int        conv_width,
                       unsigned int        conv_height,
                       unsigned int        ofm,
                       ITensorAccessorUPtr weights,
                       ITensorAccessorUPtr bias,
                       PadStrideInfo       deconv_info)
        : _conv_width(conv_width),
          _conv_height(conv_height),
          _ofm(ofm),
          _deconv_info(std::move(deconv_info)),
          _weights(std::move(weights)),
          _bias(std::move(bias))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeIdxPair input         = { s.tail_node(), 0 };
        NodeParams  common_params = { name(), s.hints().target_hint };
        return GraphBuilder::add_deconvolution_node(s.graph(), common_params, input,
                                                    Size2D(_conv_width, _conv_height), _ofm, _deconv_info,
                                                    std::move(_weights), std::move(_bias));
    }

private:
    unsigned int        _conv_width;
    unsigned int        _conv_height;
    unsigned int        _ofm;
    const PadStrideInfo _deconv_info;
    ITensorAccessorUPtr _weights;
    ITensorAccessorUPtr _bias;
};

/** Depthwise Convolution Layer */
class DepthwiseConvolutionLayer final : public ILayer
{
public:
    /** Construct a depthwise convolution layer.
     *
     * @param[in] conv_width         Convolution width.
     * @param[in] conv_height        Convolution height.
     * @param[in] weights            Accessor to get kernel weights from.
     * @param[in] bias               Accessor to get kernel bias from.
     * @param[in] conv_info          Padding and stride information.
     * @param[in] depth_multiplier   (Optional) Depth multiplier parameter.
     * @param[in] weights_quant_info (Optional) Quantization info used for weights
     * @param[in] out_quant_info     (Optional) Output quantization info
     */
    DepthwiseConvolutionLayer(unsigned int           conv_width,
                              unsigned int           conv_height,
                              ITensorAccessorUPtr    weights,
                              ITensorAccessorUPtr    bias,
                              PadStrideInfo          conv_info,
                              int                    depth_multiplier   = 1,
                              const QuantizationInfo weights_quant_info = QuantizationInfo(),
                              const QuantizationInfo out_quant_info     = QuantizationInfo())
        : _conv_width(conv_width),
          _conv_height(conv_height),
          _conv_info(std::move(conv_info)),
          _weights(std::move(weights)),
          _bias(std::move(bias)),
          _depth_multiplier(depth_multiplier),
          _weights_quant_info(std::move(weights_quant_info)),
          _out_quant_info(std::move(out_quant_info))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeIdxPair input         = { s.tail_node(), 0 };
        NodeParams  common_params = { name(), s.hints().target_hint };
        return GraphBuilder::add_depthwise_convolution_node(s.graph(), common_params,
                                                            input, Size2D(_conv_width, _conv_height), _conv_info, _depth_multiplier,
                                                            s.hints().depthwise_convolution_method_hint,
                                                            std::move(_weights), std::move(_bias), std::move(_weights_quant_info), std::move(_out_quant_info));
    }

private:
    unsigned int           _conv_width;
    unsigned int           _conv_height;
    const PadStrideInfo    _conv_info;
    ITensorAccessorUPtr    _weights;
    ITensorAccessorUPtr    _bias;
    int                    _depth_multiplier;
    const QuantizationInfo _weights_quant_info;
    const QuantizationInfo _out_quant_info;
};

/** DepthToSpace Layer */
class DepthToSpaceLayer final : public ILayer
{
public:
    /** Construct an DepthToSpace layer.
     *
     * @param[in] block_shape Block size to rearranged
     */
    DepthToSpaceLayer(int32_t block_shape)
        : _block_shape(block_shape)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_depth_to_space_node(s.graph(), common_params, input, _block_shape);
    }

private:
    int32_t _block_shape;
};

/** Dequantization Layer */
class DequantizationLayer final : public ILayer
{
public:
    /** Construct a dequantization layer.
     *
     */
    DequantizationLayer()
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_dequantization_node(s.graph(), common_params, input);
    }
};

/** DetectionOutput Layer */
class DetectionOutputLayer final : public ILayer
{
public:
    /** Construct a detection output layer.
     *
     * @param[in] sub_stream_conf  Confidence graph sub-stream.
     * @param[in] sub_stream_prior PriorBox graph sub-stream.
     * @param[in] detect_info      DetectionOutput parameters.
     */
    DetectionOutputLayer(SubStream &&sub_stream_conf, SubStream &&sub_stream_prior, const DetectionOutputLayerInfo &detect_info)
        : _ss_conf(std::move(sub_stream_conf)), _ss_prior(std::move(sub_stream_prior)), _detect_info(detect_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params  = { name(), s.hints().target_hint };
        NodeIdxPair input_loc      = { s.tail_node(), 0 };
        NodeIdxPair input_conf     = { _ss_conf.tail_node(), 0 };
        NodeIdxPair input_priorbox = { _ss_prior.tail_node(), 0 };
        return GraphBuilder::add_detection_output_node(s.graph(), common_params, input_loc, input_conf, input_priorbox, _detect_info);
    }

private:
    SubStream                _ss_conf;
    SubStream                _ss_prior;
    DetectionOutputLayerInfo _detect_info;
};
/** DetectionOutputPostProcess Layer */
class DetectionPostProcessLayer final : public ILayer
{
public:
    /** Construct a detection output layer.
     *
     * @param[in] sub_stream_class_prediction Class prediction graph sub-stream.
     * @param[in] detect_info                 DetectionOutput parameters.
     * @param[in] anchors                     Accessor to get anchors tensor data from.
     * @param[in] out_quant_info              (Optional) Output quantization info
     */
    DetectionPostProcessLayer(SubStream &&sub_stream_class_prediction, DetectionPostProcessLayerInfo detect_info, ITensorAccessorUPtr anchors,
                              const QuantizationInfo out_quant_info = QuantizationInfo())
        : _sub_stream_class_prediction(std::move(sub_stream_class_prediction)), _detect_info(detect_info), _anchors(std::move(anchors)), _out_quant_info(std::move(out_quant_info))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        ARM_COMPUTE_ERROR_ON(_anchors == nullptr);

        NodeParams  common_params          = { name(), s.hints().target_hint };
        NodeIdxPair input_box_encoding     = { s.tail_node(), 0 };
        NodeIdxPair input_class_prediction = { _sub_stream_class_prediction.tail_node(), 0 };
        return GraphBuilder::add_detection_post_process_node(s.graph(), common_params, input_box_encoding, input_class_prediction, _detect_info, std::move(_anchors), std::move(_out_quant_info));
    }

private:
    SubStream                     _sub_stream_class_prediction;
    DetectionPostProcessLayerInfo _detect_info;
    ITensorAccessorUPtr           _anchors;
    const QuantizationInfo        _out_quant_info;
};
/** Dummy Layer */
class DummyLayer final : public ILayer
{
public:
    /** Construct a dummy layer.
     *
     * @param[in] shape Output shape
     */
    DummyLayer(TensorShape shape)
        : _shape(shape)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_dummy_node(s.graph(), common_params, input, _shape);
    }

private:
    TensorShape _shape;
};

class EltwiseLayer final : public ILayer
{
public:
    /** Construct an element-wise operation layer
     *
     * @param[in] sub_stream0 First graph sub-stream
     * @param[in] sub_stream1 First graph sub-stream
     * @param[in] op          Element-wise operation to perform
     */
    EltwiseLayer(SubStream &&sub_stream0, SubStream &&sub_stream1, EltwiseOperation op)
        : _ss0(std::move(sub_stream0)), _ss1(std::move(sub_stream1)), _op(op)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input0        = { _ss0.tail_node(), 0 };
        NodeIdxPair input1        = { _ss1.tail_node(), 0 };

        return GraphBuilder::add_elementwise_node(s.graph(), common_params, input0, input1, _op);
    }

private:
    SubStream        _ss0;
    SubStream        _ss1;
    EltwiseOperation _op;
};
/** Flatten Layer */
class FlattenLayer final : public ILayer
{
public:
    /** Construct a flatten layer. */
    FlattenLayer()
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_flatten_node(s.graph(), common_params, input);
    }
};

/** Fully Connected Layer */
class FullyConnectedLayer final : public ILayer
{
public:
    /** Construct a fully connected layer.
     *
     * @param[in] num_outputs        Number of outputs.
     * @param[in] weights            Accessor to get weights from.
     * @param[in] bias               Accessor to get bias from.
     * @param[in] fc_info            (Optional) Fully connected layer metadata
     * @param[in] weights_quant_info (Optional) Weights quantization information
     * @param[in] out_quant_info     (Optional) Output quantization info
     */
    FullyConnectedLayer(unsigned int                  num_outputs,
                        ITensorAccessorUPtr           weights,
                        ITensorAccessorUPtr           bias,
                        const FullyConnectedLayerInfo fc_info            = FullyConnectedLayerInfo(),
                        const QuantizationInfo        weights_quant_info = QuantizationInfo(),
                        const QuantizationInfo        out_quant_info     = QuantizationInfo())
        : _num_outputs(num_outputs),
          _weights(std::move(weights)),
          _bias(std::move(bias)),
          _weights_ss(nullptr),
          _bias_ss(nullptr),
          _fc_info(fc_info),
          _weights_quant_info(std::move(weights_quant_info)),
          _out_quant_info(std::move(out_quant_info))
    {
    }

    /** Construct a fully connected layer.
     *
     * @param[in] num_outputs        Number of outputs.
     * @param[in] sub_stream_weights Graph sub-stream for the weights.
     * @param[in] sub_stream_bias    Graph sub-stream for the bias.
     * @param[in] fc_info            (Optional) Fully connected layer metadata
     * @param[in] weights_quant_info (Optional) Weights quantization information
     * @param[in] out_quant_info     (Optional) Output quantization info
     */
    FullyConnectedLayer(unsigned int                  num_outputs,
                        SubStream                     sub_stream_weights,
                        SubStream                     sub_stream_bias,
                        const FullyConnectedLayerInfo fc_info            = FullyConnectedLayerInfo(),
                        const QuantizationInfo        weights_quant_info = QuantizationInfo(),
                        const QuantizationInfo        out_quant_info     = QuantizationInfo())
        : _num_outputs(num_outputs),
          _weights(nullptr),
          _bias(nullptr),
          _weights_ss(std::make_unique<SubStream>(std::move(sub_stream_weights))),
          _bias_ss(std::make_unique<SubStream>(std::move(sub_stream_bias))),
          _fc_info(fc_info),
          _weights_quant_info(std::move(weights_quant_info)),
          _out_quant_info(std::move(out_quant_info))
    {
    }

    /** Create layer and add to the given stream.
     *
     * @param[in] s Stream to add layer to.
     *
     * @return ID of the created node.
     */
    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        if(_weights != nullptr)
        {
            return GraphBuilder::add_fully_connected_layer(s.graph(), common_params, input, _num_outputs,
                                                           std::move(_weights), std::move(_bias), _fc_info,
                                                           std::move(_weights_quant_info), std::move(_out_quant_info), s.hints().fast_math_hint);
        }
        else
        {
            ARM_COMPUTE_ERROR_ON(_weights_ss == nullptr);

            NodeID bias_nid = (_bias_ss == nullptr) ? EmptyNodeID : _bias_ss->tail_node();
            return GraphBuilder::add_fully_connected_layer(s.graph(), common_params, input, _num_outputs,
                                                           _weights_ss->tail_node(), bias_nid, _fc_info,
                                                           std::move(_out_quant_info), s.hints().fast_math_hint);
        }
    }

private:
    unsigned int                  _num_outputs;
    ITensorAccessorUPtr           _weights;
    ITensorAccessorUPtr           _bias;
    std::unique_ptr<SubStream>    _weights_ss;
    std::unique_ptr<SubStream>    _bias_ss;
    const FullyConnectedLayerInfo _fc_info;
    const QuantizationInfo        _weights_quant_info;
    const QuantizationInfo        _out_quant_info;
};

/** Generate Proposals Layer */
class GenerateProposalsLayer final : public ILayer
{
public:
    /** Construct a generate proposals layer.
     *
     * @param[in] ss_scores  Graph sub-stream for the scores.
     * @param[in] ss_deltas  Graph sub-stream for the deltas.
     * @param[in] ss_anchors Graph sub-stream for the anchors.
     * @param[in] info       Generate Proposals operation information.
     */
    GenerateProposalsLayer(SubStream &&ss_scores, SubStream &&ss_deltas, SubStream &&ss_anchors, GenerateProposalsInfo info)
        : _ss_scores(std::move(ss_scores)), _ss_deltas(std::move(ss_deltas)), _ss_anchors(std::move(ss_anchors)), _info(info)
    {
    }

    /** Create layer and add to the given stream.
     *
     * @param[in] s Stream to add layer to.
     *
     * @return ID of the created node.
     */
    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair scores        = { _ss_scores.tail_node(), 0 };
        NodeIdxPair deltas        = { _ss_deltas.tail_node(), 0 };
        NodeIdxPair anchors       = { _ss_anchors.tail_node(), 0 };
        return GraphBuilder::add_generate_proposals_node(s.graph(), common_params, scores, deltas, anchors, _info);
    }

private:
    SubStream             _ss_scores;
    SubStream             _ss_deltas;
    SubStream             _ss_anchors;
    GenerateProposalsInfo _info;
};

/** L2 Normalize Layer */
class L2NormalizeLayer final : public ILayer
{
public:
    /** Construct a L2 Normalize layer.
     *
     * @param[in] axis    Axis to perform normalization on
     * @param[in] epsilon Lower bound value for the normalization
     */
    L2NormalizeLayer(int axis, float epsilon)
        : _axis(axis), _epsilon(epsilon)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_l2_normalize_node(s.graph(), common_params, input, _axis, _epsilon);
    }

private:
    int   _axis;
    float _epsilon;
};

/** Normalization Layer */
class NormalizationLayer final : public ILayer
{
public:
    /** Construct a normalization layer.
     *
     * @param[in] norm_info Normalization information.
     */
    NormalizationLayer(NormalizationLayerInfo norm_info)
        : _norm_info(norm_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_normalization_node(s.graph(), common_params, input, _norm_info);
    }

private:
    NormalizationLayerInfo _norm_info;
};

/** Normalize planar YUV Layer */
class NormalizePlanarYUVLayer final : public ILayer
{
public:
    /** Construct a normalize planar YUV layer.
     *
     * @param[in] mean Accessor to get mean tensor data from.
     * @param[in] std  Accessor to get std tensor data from.
     */
    NormalizePlanarYUVLayer(ITensorAccessorUPtr mean,
                            ITensorAccessorUPtr std)
        : _mean(std::move(mean)), _std(std::move(std))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        ARM_COMPUTE_ERROR_ON(_mean == nullptr);
        ARM_COMPUTE_ERROR_ON(_std == nullptr);

        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_normalize_planar_yuv_node(s.graph(), common_params, input,
                                                           std::move(_mean), std::move(_std));
    }

private:
    ITensorAccessorUPtr _mean;
    ITensorAccessorUPtr _std;
};

/** Pad Layer */
class PadLayer final : public ILayer
{
public:
    /** Construct a pad layer.
     *
     * @param[in] padding   The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                      specifies the front and the end padding in the i-th dimension.
     * @param[in] pad_value Padding value to use. Defaults to 0.
     */
    PadLayer(PaddingList padding, PixelValue pad_value = PixelValue())
        : _padding(padding), _pad_value(pad_value)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_pad_node(s.graph(), common_params, input, _padding, _pad_value);
    }

private:
    PaddingList _padding;
    PixelValue  _pad_value;
};

/** Permute Layer */
class PermuteLayer final : public ILayer
{
public:
    /** Construct a permute layer.
     *
     * @param[in] perm   Permutation vector.
     * @param[in] layout (Optional) Data layout to assign to permuted tensor.
     *                   If UNKNOWN then the input's layout will be used.
     */
    PermuteLayer(PermutationVector perm, DataLayout layout = DataLayout::UNKNOWN)
        : _perm(perm), _layout(layout)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_permute_node(s.graph(), common_params, input, _perm, _layout);
    }

private:
    PermutationVector _perm;
    DataLayout        _layout;
};

/** Pooling Layer */
class PoolingLayer final : public ILayer
{
public:
    /** Construct a pooling layer.
     *
     * @param[in] pool_info Pooling information.
     */
    PoolingLayer(PoolingLayerInfo pool_info)
        : _pool_info(pool_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_pooling_node(s.graph(), common_params, input, _pool_info);
    }

private:
    PoolingLayerInfo _pool_info;
};

/** PRelu Layer */
class PReluLayer final : public ILayer
{
public:
    /** Construct an PRelu operation layer
     *
     * @param[in] sub_stream0 First graph sub-stream
     * @param[in] sub_stream1 First graph sub-stream
     */
    PReluLayer(SubStream &&sub_stream0, SubStream &&sub_stream1)
        : _ss0(std::move(sub_stream0)), _ss1(std::move(sub_stream1))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { _ss0.tail_node(), 0 };
        NodeIdxPair alpha         = { _ss1.tail_node(), 0 };

        return GraphBuilder::add_prelu_node(s.graph(), common_params, input, alpha);
    }

private:
    SubStream _ss0;
    SubStream _ss1;
};

/** Print Layer */
class PrintLayer final : public ILayer
{
public:
    /** Construct a print layer.
     *
     * Example usage to locally dequantize and print a tensor:
     *
     * Tensor *output = new Tensor();
     * const auto transform = [output](ITensor *input)
     * {
     *     output->allocator()->init(*input->info());
     *     output->info()->set_data_type(DataType::F32);
     *     output->allocator()->allocate();
     *
     *     Window win;
     *     win.use_tensor_dimensions(input->info()->tensor_shape());
     *     Iterator in(input, win);
     *     Iterator out(output, win);
     *     execute_window_loop(win, [&](const Coordinates &)
     *     {
     *         *(reinterpret_cast<float *>(out.ptr())) = dequantize_qasymm8(*in.ptr(), input->info()->quantization_info().uniform());
     *     }, in, out);
     *
     *     return output;
     * };
     *
     * graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info), get_input_accessor(common_params, nullptr, false))
     *       << ...
     *       << \\ CNN Layers
     *       << ...
     *       << PrintLayer(std::cout, IOFormatInfo(), transform)
     *       << ...
     *       << OutputLayer(get_output_accessor(common_params, 5));
     *
     * @param[in] stream      Output stream.
     * @param[in] format_info (Optional) Format info.
     * @param[in] transform   (Optional) Input transform function.
     */
    PrintLayer(std::ostream &stream, const IOFormatInfo &format_info = IOFormatInfo(), const std::function<ITensor *(ITensor *)> transform = nullptr)
        : _stream(stream), _format_info(format_info), _transform(transform)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_print_node(s.graph(), common_params, input, _stream, _format_info, _transform);
    }

private:
    std::ostream                             &_stream;
    const IOFormatInfo                       &_format_info;
    const std::function<ITensor *(ITensor *)> _transform;
};

/** PriorBox Layer */
class PriorBoxLayer final : public ILayer
{
public:
    /** Construct a priorbox layer.
     *
     * @param[in] sub_stream First graph sub-stream
     * @param[in] prior_info PriorBox parameters.
     */
    PriorBoxLayer(SubStream &&sub_stream, const PriorBoxLayerInfo &prior_info)
        : _ss(std::move(sub_stream)), _prior_info(prior_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input0        = { s.tail_node(), 0 };
        NodeIdxPair input1        = { _ss.tail_node(), 0 };
        return GraphBuilder::add_priorbox_node(s.graph(), common_params, input0, input1, _prior_info);
    }

private:
    SubStream         _ss;
    PriorBoxLayerInfo _prior_info;
};

/** Quantization Layer */
class QuantizationLayer final : public ILayer
{
public:
    /** Construct a quantization layer.
     *
     * @param[in] out_quant_info Output tensor quantization info
     */
    QuantizationLayer(QuantizationInfo out_quant_info)
        : _out_quant_info(out_quant_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_quantization_node(s.graph(), common_params, input, _out_quant_info);
    }

private:
    QuantizationInfo _out_quant_info;
};

/** Reduction Layer */
class ReductionLayer final : public ILayer
{
public:
    /** Construct a reduction layer.
     *
     * @param[in] op        Reduction operation
     * @param[in] axis      Reduction axis
     * @param[in] keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
     */
    ReductionLayer(ReductionOperation op, unsigned int axis, bool keep_dims)
        : _op(op), _axis(axis), _keep_dims(keep_dims)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_reduction_operation_node(s.graph(), common_params, input, _op, _axis, _keep_dims);
    }

private:
    ReductionOperation _op;
    unsigned int       _axis;
    bool               _keep_dims;
};

/** Reorg Layer */
class ReorgLayer final : public ILayer
{
public:
    /** Construct a reorg layer.
     *
     * @param[in] stride Stride value to use for reorganizing the values in the output tensor.
     *                   It defines the spatial distance between 2 consecutive pixels in the x and y direction
     */
    ReorgLayer(int stride)
        : _stride(stride)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_reorg_node(s.graph(), common_params, input, _stride);
    }

private:
    int _stride;
};

/** Reshape Layer */
class ReshapeLayer final : public ILayer
{
public:
    /** Construct a reshape layer.
     *
     * @param[in] shape Target shape.
     */
    ReshapeLayer(TensorShape shape)
        : _shape(shape)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_reshape_node(s.graph(), common_params, input, _shape);
    }

private:
    TensorShape _shape;
};

/** Resize Layer */
class ResizeLayer final : public ILayer
{
public:
    ResizeLayer(InterpolationPolicy policy, float width_scale, float height_scale)
        : _policy(policy), _width_scale(width_scale), _height_scale(height_scale)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_resize_node(s.graph(), common_params, input, _policy, _width_scale, _height_scale);
    }

private:
    InterpolationPolicy _policy;
    float               _width_scale;
    float               _height_scale;
};

/** ROIAlign Layer */
class ROIAlignLayer final : public ILayer
{
public:
    /** Construct a RoiAlign layer.
     *
     * @param[in] sub_stream_input Graph sub-stream for the input
     * @param[in] sub_stream_rois  Graph sub-stream for the rois
     * @param[in] pool_info        Pooling information.
     */
    ROIAlignLayer(SubStream &&sub_stream_input, SubStream &&sub_stream_rois, ROIPoolingLayerInfo pool_info)
        : _ss_input(sub_stream_input), _ss_rois(sub_stream_rois), _pool_info(pool_info)
    {
    }

    /** Prevent instances of this class from being copy constructed */
    ROIAlignLayer(const ROIAlignLayer &) = delete;
    /** Prevent instances of this class from being copied */
    ROIAlignLayer &operator=(const ROIAlignLayer &) = delete;

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { _ss_input.tail_node(), 0 };
        NodeIdxPair rois          = { _ss_rois.tail_node(), 0 };
        return GraphBuilder::add_roi_align_node(s.graph(), common_params, input, rois, _pool_info);
    }

private:
    SubStream           _ss_input;
    SubStream           _ss_rois;
    ROIPoolingLayerInfo _pool_info;
};

/** Scale Layer */
class ScaleLayer final : public ILayer
{
public:
    /** Construct a scale layer.
     *
     * @param[in] mul_w Accessor to get mul weight from.
     * @param[in] add_w Accessor to get add weight from.
     */
    ScaleLayer(ITensorAccessorUPtr mul_w,
               ITensorAccessorUPtr add_w)
        : _mul_w(std::move(mul_w)), _add_w(std::move(add_w))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_scale_layer(s.graph(), common_params, input, std::move(_mul_w), std::move(_add_w));
    }

private:
    ITensorAccessorUPtr _mul_w;
    ITensorAccessorUPtr _add_w;
};

/** Slice Layer */
class SliceLayer final : public ILayer
{
public:
    /** Construct a slice layer.
     *
     * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     */
    SliceLayer(Coordinates &starts, Coordinates &ends)
        : _starts(starts), _ends(ends)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_slice_node(s.graph(), common_params, input, _starts, _ends);
    }

private:
    Coordinates _starts;
    Coordinates _ends;
};

/** Softmax Layer */
class SoftmaxLayer final : public ILayer
{
public:
    /** Construct a softmax layer.
     *
     * @param[in] beta (Optional) Beta value. Default 1.0.
     */
    SoftmaxLayer(float beta = 1.0f)
        : _beta(beta)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_softmax_node(s.graph(), common_params, input, _beta);
    }

private:
    float _beta;
};

/** Stack Layer */
class StackLayer final : public ILayer
{
public:
    /** Construct a concatenation layer
     *
     * @param[in] sub_stream1      First graph branch
     * @param[in] sub_stream2      Second graph branch
     * @param[in] rest_sub_streams Rest sub-graph branches
     */
    template <typename... Ts>
    StackLayer(SubStream &&sub_stream1, SubStream &&sub_stream2, Ts &&... rest_sub_streams)
        : _sub_streams(), _axis(0)
    {
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream1)));
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream2)));

        utility::for_each([&](SubStream && sub_stream)
        {
            _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream)));
        },
        std::move(rest_sub_streams)...);
    }
    /** Construct a concatenation layer
     *
     * @param[in] axis             Stack layer axis along which to stack the inputs
     * @param[in] sub_stream1      First graph branch
     * @param[in] sub_stream2      Second graph branch
     * @param[in] rest_sub_streams Rest sub-graph branches
     */
    template <typename... Ts>
    StackLayer(int axis, SubStream &&sub_stream1, SubStream &&sub_stream2, Ts &&... rest_sub_streams)
        : _sub_streams(), _axis(axis)
    {
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream1)));
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream2)));

        utility::for_each([&](SubStream && sub_stream)
        {
            _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream)));
        },
        std::move(rest_sub_streams)...);
    }
    /** Construct a concat layer
     *
     * @param[in] sub_stream Sub-stream
     */
    template <typename... Ts>
    StackLayer(SubStream &&sub_stream)
        : _sub_streams(), _axis(0)
    {
        _sub_streams.push_back(std::make_unique<SubStream>(std::move(sub_stream)));
    }
    NodeID create_layer(IStream &s) override
    {
        NodeID     nid           = EmptyNodeID;
        NodeParams common_params = { name(), s.hints().target_hint };
        if(_sub_streams.size() == 1 && _sub_streams.at(0) != nullptr)
        {
            nid = _sub_streams[0]->tail_node();
        }
        else
        {
            // Collect tail nodes and stack
            std::vector<NodeIdxPair> nodes;
            for(auto &ss : _sub_streams)
            {
                if(ss && (ss->tail_node() != EmptyNodeID))
                {
                    const auto tail_node = s.graph().node(ss->tail_node());
                    if(tail_node != nullptr && tail_node->type() != NodeType::Output)
                    {
                        nodes.push_back({ ss->tail_node(), 0 });
                    }
                }
            }
            nid = GraphBuilder::add_stack_node(s.graph(), common_params, nodes, _axis);
        }
        return nid;
    }

private:
    std::vector<std::unique_ptr<SubStream>> _sub_streams;
    int                                     _axis;
};

/** StridedSlice Layer */
class StridedSliceLayer final : public ILayer
{
public:
    /** Construct a strided slice layer.
     *
     * @param[in] starts             The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends               The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] strides            The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] strided_slice_info Contains masks for the starts, ends and strides
     */
    StridedSliceLayer(Coordinates &starts, Coordinates &ends, BiStrides &strides, StridedSliceLayerInfo strided_slice_info)
        : _starts(starts), _ends(ends), _strides(strides), _info(strided_slice_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_strided_slice_node(s.graph(), common_params, input, _starts, _ends, _strides, _info);
    }

private:
    Coordinates           _starts;
    Coordinates           _ends;
    BiStrides             _strides;
    StridedSliceLayerInfo _info;
};

/** YOLO Layer */
class YOLOLayer final : public ILayer
{
public:
    /** Construct a YOLO layer.
     *
     * @param[in] act_info Activation info
     */
    YOLOLayer(ActivationLayerInfo act_info)
        : _act_info(act_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { name(), s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_yolo_node(s.graph(), common_params, input, _act_info);
    }

private:
    ActivationLayerInfo _act_info;
};
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_LAYERS_H */
