/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/graph/backends/NEON/NEFunctionFactory.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/backends/FunctionHelpers.h"
#include "arm_compute/graph/backends/Utils.h"
#include "arm_compute/graph/nodes/Nodes.h"
#include "arm_compute/runtime/CPP/CPPFunctions.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "support/Cast.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Target specific information structure used to pass information to the layer templates */
struct NETargetInfo
{
    using TensorType         = arm_compute::ITensor;
    using SrcTensorType      = const arm_compute::ITensor;
    using TensorConcreteType = arm_compute::Tensor;
    static Target TargetType;
};

Target NETargetInfo::TargetType = Target::NEON;

/** Collection of Neon convolution functions */
struct NEConvolutionLayerFunctions
{
    using GenericConvolutionLayer  = NEConvolutionLayer;
    using GEMMConvolutionLayer     = NEGEMMConvolutionLayer;
    using DirectConvolutionLayer   = NEDirectConvolutionLayer;
    using WinogradConvolutionLayer = NEWinogradConvolutionLayer;
};

/** Collection of Neon element-wise functions */
struct NEEltwiseFunctions
{
    using Addition       = NEArithmeticAddition;
    using Subtraction    = NEArithmeticSubtraction;
    using Multiplication = NEPixelWiseMultiplication;
    using Maximum        = NEElementwiseMax;
};

/** Collection of Neon unary element-wise functions */
struct NEUnaryEltwiseFunctions
{
    using Exp = NEExpLayer;
};

/** Function and tensor types to be used inside a Neon fused convolution/batch normalization layer */
struct NEFusedLayerTypes
{
    using ConvolutionLayer          = NEConvolutionLayer;
    using DepthwiseConvolutionLayer = NEDepthwiseConvolutionLayer;
    using FuseBatchNormalization    = NEFuseBatchNormalization;
};

namespace detail
{
template <>
std::unique_ptr<IFunction> create_normalization_layer<NENormalizationLayer, NETargetInfo>(NormalizationLayerNode &node, GraphContext &ctx)
{
    validate_node<NETargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    NETargetInfo::TensorType    *input     = get_backing_tensor<NETargetInfo>(node.input(0));
    NETargetInfo::TensorType    *output    = get_backing_tensor<NETargetInfo>(node.output(0));
    const NormalizationLayerInfo norm_info = node.normalization_info();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<NENormalizationLayer>(get_memory_manager(ctx, NETargetInfo::TargetType));
    func->configure(input, output, norm_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << NETargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Normalization info: " << norm_info.type()
                               << std::endl);

    return std::move(func);
}
} // namespace detail

std::unique_ptr<IFunction> NEFunctionFactory::create(INode *node, GraphContext &ctx)
{
    if(node == nullptr)
    {
        return nullptr;
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::ActivationLayer:
            return detail::create_activation_layer<NEActivationLayer, NETargetInfo>(*polymorphic_downcast<ActivationLayerNode *>(node));
        case NodeType::ArgMinMaxLayer:
            return detail::create_arg_min_max_layer<NEArgMinMaxLayer, NETargetInfo>(*polymorphic_downcast<ArgMinMaxLayerNode *>(node));
        case NodeType::BatchNormalizationLayer:
            return detail::create_batch_normalization_layer<NEBatchNormalizationLayer, NETargetInfo>(*polymorphic_downcast<BatchNormalizationLayerNode *>(node));
        case NodeType::ChannelShuffleLayer:
            return detail::create_channel_shuffle_layer<NEChannelShuffleLayer, NETargetInfo>(*polymorphic_downcast<ChannelShuffleLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return detail::create_convolution_layer<NEConvolutionLayerFunctions, NETargetInfo>(*polymorphic_downcast<ConvolutionLayerNode *>(node), ctx);
        case NodeType::DepthToSpaceLayer:
            return detail::create_depth_to_space_layer<NEDepthToSpaceLayer, NETargetInfo>(*polymorphic_downcast<DepthToSpaceLayerNode *>(node));
        case NodeType::DeconvolutionLayer:
            return detail::create_deconvolution_layer<NEDeconvolutionLayer, NETargetInfo>(*polymorphic_downcast<DeconvolutionLayerNode *>(node), ctx);
        case NodeType::ConcatenateLayer:
            return detail::create_concatenate_layer<NEConcatenateLayer, NETargetInfo>(*polymorphic_downcast<ConcatenateLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::create_depthwise_convolution_layer<NEDepthwiseConvolutionLayer, NETargetInfo>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::DequantizationLayer:
            return detail::create_dequantization_layer<NEDequantizationLayer, NETargetInfo>(*polymorphic_downcast<DequantizationLayerNode *>(node));
        case NodeType::DetectionOutputLayer:
            return detail::create_detection_output_layer<CPPDetectionOutputLayer, NETargetInfo>(*polymorphic_downcast<DetectionOutputLayerNode *>(node));
        case NodeType::DetectionPostProcessLayer:
            return detail::create_detection_post_process_layer<NEDetectionPostProcessLayer, NETargetInfo>(*polymorphic_downcast<DetectionPostProcessLayerNode *>(node));
        case NodeType::EltwiseLayer:
            return detail::create_eltwise_layer<NEEltwiseFunctions, NETargetInfo>(*polymorphic_downcast<EltwiseLayerNode *>(node));
        case NodeType::UnaryEltwiseLayer:
            return detail::create_unary_eltwise_layer<NEUnaryEltwiseFunctions, NETargetInfo>(*polymorphic_downcast<UnaryEltwiseLayerNode *>(node));
        case NodeType::FlattenLayer:
            return detail::create_flatten_layer<NEFlattenLayer, NETargetInfo>(*polymorphic_downcast<FlattenLayerNode *>(node));
        case NodeType::FullyConnectedLayer:
            return detail::create_fully_connected_layer<NEFullyConnectedLayer, NETargetInfo>(*polymorphic_downcast<FullyConnectedLayerNode *>(node), ctx);
        case NodeType::FusedConvolutionBatchNormalizationLayer:
            return detail::create_fused_convolution_batch_normalization_layer<NEFusedLayerTypes, NETargetInfo>(*polymorphic_downcast<FusedConvolutionBatchNormalizationNode *>(node), ctx);
        case NodeType::FusedDepthwiseConvolutionBatchNormalizationLayer:
            return detail::create_fused_depthwise_convolution_batch_normalization_layer<NEFusedLayerTypes, NETargetInfo>(*polymorphic_downcast<FusedDepthwiseConvolutionBatchNormalizationNode *>(node), ctx);
        case NodeType::L2NormalizeLayer:
            return detail::create_l2_normalize_layer<NEL2NormalizeLayer, NETargetInfo>(*polymorphic_downcast<L2NormalizeLayerNode *>(node), ctx);
        case NodeType::NormalizationLayer:
            return detail::create_normalization_layer<NENormalizationLayer, NETargetInfo>(*polymorphic_downcast<NormalizationLayerNode *>(node), ctx);
        case NodeType::PadLayer:
            return detail::create_pad_layer<NEPadLayer, NETargetInfo>(*polymorphic_downcast<PadLayerNode *>(node));
        case NodeType::PermuteLayer:
            return detail::create_permute_layer<NEPermute, NETargetInfo>(*polymorphic_downcast<PermuteLayerNode *>(node));
        case NodeType::PoolingLayer:
            return detail::create_pooling_layer<NEPoolingLayer, NETargetInfo>(*polymorphic_downcast<PoolingLayerNode *>(node));
        case NodeType::PReluLayer:
            return detail::create_prelu_layer<NEPReluLayer, NETargetInfo>(*polymorphic_downcast<PReluLayerNode *>(node));
        case NodeType::PrintLayer:
            return detail::create_print_layer<NETargetInfo>(*polymorphic_downcast<PrintLayerNode *>(node));
        case NodeType::PriorBoxLayer:
            return detail::create_priorbox_layer<NEPriorBoxLayer, NETargetInfo>(*polymorphic_downcast<PriorBoxLayerNode *>(node));
        case NodeType::QuantizationLayer:
            return detail::create_quantization_layer<NEQuantizationLayer, NETargetInfo>(*polymorphic_downcast<QuantizationLayerNode *>(node));
        case NodeType::ReductionOperationLayer:
            return detail::create_reduction_operation_layer<NEReductionOperation, NETargetInfo>(*polymorphic_downcast<ReductionLayerNode *>(node), ctx);
        case NodeType::ReorgLayer:
            return detail::create_reorg_layer<NEReorgLayer, NETargetInfo>(*polymorphic_downcast<ReorgLayerNode *>(node));
        case NodeType::ReshapeLayer:
            return detail::create_reshape_layer<NEReshapeLayer, NETargetInfo>(*polymorphic_downcast<ReshapeLayerNode *>(node));
        case NodeType::ResizeLayer:
            return detail::create_resize_layer<NEScale, NETargetInfo>(*polymorphic_downcast<ResizeLayerNode *>(node));
        case NodeType::SliceLayer:
            return detail::create_slice_layer<NESlice, NETargetInfo>(*polymorphic_downcast<SliceLayerNode *>(node));
        case NodeType::SoftmaxLayer:
            return detail::create_softmax_layer<NESoftmaxLayer, NETargetInfo>(*polymorphic_downcast<SoftmaxLayerNode *>(node), ctx);
        case NodeType::StackLayer:
            return detail::create_stack_layer<NEStackLayer, NETargetInfo>(*polymorphic_downcast<StackLayerNode *>(node));
        case NodeType::StridedSliceLayer:
            return detail::create_strided_slice_layer<NEStridedSlice, NETargetInfo>(*polymorphic_downcast<StridedSliceLayerNode *>(node));
        default:
            return nullptr;
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
