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
#include "arm_compute/graph/backends/CL/CLFunctionFactory.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/backends/FunctionHelpers.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Target specific information structure used to pass information to the layer templates */
struct CLTargetInfo
{
    using TensorType = arm_compute::ICLTensor;
    static Target TargetType;
};

Target CLTargetInfo::TargetType = Target::CL;

/** Collection of CL convolution functions */
struct CLConvolutionLayerFunctions
{
    using GenericConvolutionLayer  = CLConvolutionLayer;
    using GEMMConvolutionLayer     = CLGEMMConvolutionLayer;
    using DirectConvolutionLayer   = CLDirectConvolutionLayer;
    using WinogradConvolutionLayer = CLWinogradConvolutionLayer;
};

/** Collection of CL depthwise convolution functions */
struct CLDepthwiseConvolutionLayerFunctions
{
    using GenericDepthwiseConvolutionLayer = CLDepthwiseConvolutionLayer;
    using DepthwiseConvolutionLayer3x3     = CLDepthwiseConvolutionLayer3x3;
};

/** Collection of CL element-wise functions */
struct CLEltwiseFunctions
{
    using Addition       = CLArithmeticAddition;
    using Subtraction    = CLArithmeticSubtraction;
    using Multiplication = CLPixelWiseMultiplication;
};

std::unique_ptr<IFunction> CLFunctionFactory::create(INode *node, GraphContext &ctx)
{
    if(node == nullptr)
    {
        return nullptr;
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::ActivationLayer:
            return detail::create_activation_layer<CLActivationLayer, CLTargetInfo>(*polymorphic_downcast<ActivationLayerNode *>(node));
        case NodeType::BatchNormalizationLayer:
            return detail::create_batch_normalization_layer<CLBatchNormalizationLayer, CLTargetInfo>(*polymorphic_downcast<BatchNormalizationLayerNode *>(node));
        case NodeType::ChannelShuffleLayer:
            return detail::create_channel_shuffle_layer<CLChannelShuffleLayer, CLTargetInfo>(*polymorphic_downcast<ChannelShuffleLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return detail::create_convolution_layer<CLConvolutionLayerFunctions, CLTargetInfo>(*polymorphic_downcast<ConvolutionLayerNode *>(node), ctx);
        case NodeType::DeconvolutionLayer:
            return detail::create_deconvolution_layer<CLDeconvolutionLayer, CLTargetInfo>(*polymorphic_downcast<DeconvolutionLayerNode *>(node), ctx);
        case NodeType::ConcatenateLayer:
            return detail::create_concatenate_layer<CLConcatenateLayer, CLTargetInfo>(*polymorphic_downcast<ConcatenateLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::create_depthwise_convolution_layer<CLDepthwiseConvolutionLayerFunctions, CLTargetInfo>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::EltwiseLayer:
            return detail::create_eltwise_layer<CLEltwiseFunctions, CLTargetInfo>(*polymorphic_downcast<EltwiseLayerNode *>(node));
        case NodeType::FlattenLayer:
            return detail::create_flatten_layer<CLFlattenLayer, CLTargetInfo>(*polymorphic_downcast<FlattenLayerNode *>(node));
        case NodeType::FullyConnectedLayer:
            return detail::create_fully_connected_layer<CLFullyConnectedLayer, CLTargetInfo>(*polymorphic_downcast<FullyConnectedLayerNode *>(node), ctx);
        case NodeType::NormalizationLayer:
            return detail::create_normalization_layer<CLNormalizationLayer, CLTargetInfo>(*polymorphic_downcast<NormalizationLayerNode *>(node), ctx);
        case NodeType::NormalizePlanarYUVLayer:
            return detail::create_normalize_planar_yuv_layer<CLNormalizePlanarYUVLayer, CLTargetInfo>(*polymorphic_downcast<NormalizePlanarYUVLayerNode *>(node));
        case NodeType::PermuteLayer:
            return detail::create_permute_layer<CLPermute, CLTargetInfo>(*polymorphic_downcast<PermuteLayerNode *>(node));
        case NodeType::PoolingLayer:
            return detail::create_pooling_layer<CLPoolingLayer, CLTargetInfo>(*polymorphic_downcast<PoolingLayerNode *>(node));
        case NodeType::ReorgLayer:
            return detail::create_reorg_layer<CLReorgLayer, CLTargetInfo>(*polymorphic_downcast<ReorgLayerNode *>(node));
        case NodeType::ReshapeLayer:
            return detail::create_reshape_layer<CLReshapeLayer, CLTargetInfo>(*polymorphic_downcast<ReshapeLayerNode *>(node));
        case NodeType::ResizeLayer:
            return detail::create_resize_layer<CLScale, CLTargetInfo>(*polymorphic_downcast<ResizeLayerNode *>(node));
        case NodeType::SliceLayer:
            return detail::create_slice_layer<CLSlice, CLTargetInfo>(*polymorphic_downcast<SliceLayerNode *>(node));
        case NodeType::SoftmaxLayer:
            return detail::create_softmax_layer<CLSoftmaxLayer, CLTargetInfo>(*polymorphic_downcast<SoftmaxLayerNode *>(node), ctx);
        case NodeType::YOLOLayer:
            return detail::create_yolo_layer<CLYOLOLayer, CLTargetInfo>(*polymorphic_downcast<YOLOLayerNode *>(node), ctx);
        default:
            return nullptr;
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
