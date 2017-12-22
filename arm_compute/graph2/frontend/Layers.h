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
#ifndef __ARM_COMPUTE_GRAPH2_LAYERS_H__
#define __ARM_COMPUTE_GRAPH2_LAYERS_H__

#include "arm_compute/graph2/GraphBuilder.h"
#include "arm_compute/graph2/Types.h"
#include "arm_compute/graph2/frontend/ILayer.h"
#include "arm_compute/graph2/frontend/IStream.h"
#include "arm_compute/graph2/frontend/SubStream.h"

#include "arm_compute/core/utils/misc/Utility.h"

#include <memory>
#include <string>

namespace arm_compute
{
namespace graph2
{
namespace frontend
{
/** Input Layer */
class InputLayer final : public ILayer
{
public:
    InputLayer(TensorDescriptor desc, ITensorAccessorUPtr accessor)
        : _desc(desc), _accessor(std::move(accessor))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams common_params = { "", s.hints().target_hint };
        return GraphBuilder::add_input_node(s.graph(), common_params, _desc, std::move(_accessor));
    }

private:
    TensorDescriptor    _desc;
    ITensorAccessorUPtr _accessor;
};

/** Output Layer */
class OutputLayer final : public ILayer
{
public:
    OutputLayer(ITensorAccessorUPtr accessor)
        : _accessor(std::move(accessor))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_output_node(s.graph(), common_params, input, std::move(_accessor));
    }

private:
    ITensorAccessorUPtr _accessor;
};

/** Activation Layer */
class ActivationLayer final : public ILayer
{
public:
    ActivationLayer(ActivationLayerInfo act_info)
        : _act_info(act_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_activation_node(s.graph(), common_params, input, _act_info);
    }

private:
    ActivationLayerInfo _act_info;
};

/** Batchnormalization Layer */
class BatchNormalizationLayer final : public ILayer
{
public:
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

        NodeParams  common_params = { "", s.hints().target_hint };
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

/** Convolution Layer */
class ConvolutionLayer final : public ILayer
{
public:
    ConvolutionLayer(unsigned int        conv_width,
                     unsigned int        conv_height,
                     unsigned int        ofm,
                     ITensorAccessorUPtr weights,
                     ITensorAccessorUPtr bias,
                     PadStrideInfo       conv_info,
                     unsigned int        num_groups = 1)
        : _conv_width(conv_width),
          _conv_height(conv_height),
          _ofm(ofm),
          _conv_info(std::move(conv_info)),
          _num_groups(num_groups),
          _weights(std::move(weights)),
          _bias(std::move(bias))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        ARM_COMPUTE_UNUSED(_num_groups);
        NodeIdxPair input         = { s.tail_node(), 0 };
        NodeParams  common_params = { "", s.hints().target_hint };
        return GraphBuilder::add_convolution_node(s.graph(), common_params, input,
                                                  Size2D(_conv_width, _conv_height), _ofm, _conv_info,
                                                  s.hints().convolution_method_hint,
                                                  std::move(_weights), std::move(_bias));
    }

private:
    unsigned int        _conv_width;
    unsigned int        _conv_height;
    unsigned int        _ofm;
    const PadStrideInfo _conv_info;
    unsigned int        _num_groups;
    ITensorAccessorUPtr _weights;
    ITensorAccessorUPtr _bias;
};

/** Depthwise Convolution Layer */
class DepthwiseConvolutionLayer final : public ILayer
{
public:
    DepthwiseConvolutionLayer(unsigned int        conv_width,
                              unsigned int        conv_height,
                              ITensorAccessorUPtr weights,
                              ITensorAccessorUPtr bias,
                              PadStrideInfo       conv_info)
        : _conv_width(conv_width),
          _conv_height(conv_height),
          _conv_info(std::move(conv_info)),
          _weights(std::move(weights)),
          _bias(std::move(bias))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeIdxPair input         = { s.tail_node(), 0 };
        NodeParams  common_params = { "", s.hints().target_hint };
        return GraphBuilder::add_depthwise_convolution_node(s.graph(), common_params,
                                                            input, Size2D(_conv_width, _conv_height), _conv_info,
                                                            s.hints().depthwise_convolution_method_hint,
                                                            std::move(_weights), std::move(_bias));
    }

private:
    unsigned int        _conv_width;
    unsigned int        _conv_height;
    const PadStrideInfo _conv_info;
    ITensorAccessorUPtr _weights;
    ITensorAccessorUPtr _bias;
};

/** Flatten Layer */
class FlattenLayer final : public ILayer
{
public:
    FlattenLayer()
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_flatten_node(s.graph(), common_params, input);
    }
};

/** Fully Connected Layer */
class FullyConnectedLayer final : public ILayer
{
public:
    FullyConnectedLayer(unsigned int        num_outputs,
                        ITensorAccessorUPtr weights,
                        ITensorAccessorUPtr bias)
        : _num_outputs(num_outputs), _weights(std::move(weights)), _bias(std::move(bias))
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_fully_connected_layer(s.graph(), common_params, input, _num_outputs,
                                                       std::move(_weights), std::move(_bias));
    }

private:
    unsigned int        _num_outputs;
    ITensorAccessorUPtr _weights;
    ITensorAccessorUPtr _bias;
};

/** Normalization Layer */
class NormalizationLayer final : public ILayer
{
public:
    NormalizationLayer(NormalizationLayerInfo norm_info)
        : _norm_info(norm_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_normalization_node(s.graph(), common_params, input, _norm_info);
    }

private:
    NormalizationLayerInfo _norm_info;
};

/** Pooling Layer */
class PoolingLayer final : public ILayer
{
public:
    PoolingLayer(PoolingLayerInfo pool_info)
        : _pool_info(pool_info)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_pooling_node(s.graph(), common_params, input, _pool_info);
    }

private:
    PoolingLayerInfo _pool_info;
};

/** Reshape Layer */
class ReshapeLayer final : public ILayer
{
public:
    ReshapeLayer(TensorShape shape)
        : _shape(shape)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_reshape_node(s.graph(), common_params, input, _shape);
    }

private:
    TensorShape _shape;
};

/** Softmax Layer */
class SoftmaxLayer final : public ILayer
{
public:
    SoftmaxLayer(float beta = 1.0f)
        : _beta(beta)
    {
    }

    NodeID create_layer(IStream &s) override
    {
        NodeParams  common_params = { "", s.hints().target_hint };
        NodeIdxPair input         = { s.tail_node(), 0 };
        return GraphBuilder::add_softmax_node(s.graph(), common_params, input, _beta);
    }

private:
    float _beta;
};

/** Branch Layer */
class BranchLayer final : public ILayer
{
public:
    /** Default Constructor
     *
     * @param[in] merge_method     Branch merging method
     * @param[in] sub_stream1      First graph branch
     * @param[in] sub_stream2      Second graph branch
     * @param[in] rest_sub_streams Rest sub-graph branches
     */
    template <typename... Ts>
    BranchLayer(BranchMergeMethod merge_method, SubStream &&sub_stream1, SubStream &&sub_stream2, Ts &&... rest_sub_streams)
        : _branch_merge_method(merge_method), _sub_streams()
    {
        _sub_streams.push_back(arm_compute::support::cpp14::make_unique<SubStream>(std::move(sub_stream1)));
        _sub_streams.push_back(arm_compute::support::cpp14::make_unique<SubStream>(std::move(sub_stream2)));

        utility::for_each([&](SubStream && sub_stream)
        {
            _sub_streams.push_back(arm_compute::support::cpp14::make_unique<SubStream>(std::move(sub_stream)));
        },
        std::move(rest_sub_streams)...);
    }
    /** Default Constructor
     *
     * @param[in] sub_stream Sub-stream
     */
    template <typename... Ts>
    BranchLayer(SubStream &&sub_stream)
        : _branch_merge_method(BranchMergeMethod::DEPTH_CONCATENATE), _sub_streams()
    {
        _sub_streams.push_back(arm_compute::support::cpp14::make_unique<SubStream>(std::move(sub_stream)));
    }
    NodeID create_layer(IStream &s) override
    {
        NodeID     nid           = EmptyNodeID;
        NodeParams common_params = { "", s.hints().target_hint };
        if(_sub_streams.size() == 1 && _sub_streams.at(0) != nullptr)
        {
            nid = _sub_streams[0]->tail_node();
        }
        else if(_branch_merge_method == BranchMergeMethod::DEPTH_CONCATENATE)
        {
            // Collect tail nodes and perform DepthConcatenate
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
            nid = GraphBuilder::add_depth_concatenate_node(s.graph(), common_params, nodes);
        }
        else
        {
            ARM_COMPUTE_ERROR_ON(_sub_streams.size() != 2);
            NodeIdxPair input0 = { _sub_streams[0]->tail_node(), 0 };
            NodeIdxPair input1 = { _sub_streams[1]->tail_node(), 0 };
            nid                = GraphBuilder::add_elementwise_node(s.graph(), common_params, input0, input1, EltwiseOperation::ADD);
        }
        return nid;
    }

private:
    BranchMergeMethod                       _branch_merge_method;
    std::vector<std::unique_ptr<SubStream>> _sub_streams;
};
} // namespace frontend
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_LAYERS_H__ */
