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
#include "arm_compute/graph/backends/NEON/NENodeValidator.h"

#include "arm_compute/graph/backends/ValidateHelpers.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/runtime/CPP/CPPFunctions.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
Status NENodeValidator::validate(INode *node)
{
    if(node == nullptr)
    {
        return Status{};
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::BoundingBoxTransformLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : BoundingBoxTransformLayer");
        case NodeType::ChannelShuffleLayer:
            return detail::validate_channel_shuffle_layer<NEChannelShuffleLayer>(*polymorphic_downcast<ChannelShuffleLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return detail::validate_convolution_layer<NEConvolutionLayer,
                   NEDirectConvolutionLayer,
                   NEGEMMConvolutionLayer,
                   NEWinogradConvolutionLayer>(*polymorphic_downcast<ConvolutionLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::validate_depthwise_convolution_layer<NEDepthwiseConvolutionLayer,
                   NEDepthwiseConvolutionLayer3x3>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::DetectionOutputLayer:
            return detail::validate_detection_output_layer<CPPDetectionOutputLayer>(*polymorphic_downcast<DetectionOutputLayerNode *>(node));
        case NodeType::GenerateProposalsLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : GenerateProposalsLayer");
        case NodeType::NormalizePlanarYUVLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : NormalizePlanarYUVLayer");
        case NodeType::PadLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : PadLayer");
        case NodeType::PermuteLayer:
            return detail::validate_permute_layer<NEPermute>(*polymorphic_downcast<PermuteLayerNode *>(node));
        case NodeType::PriorBoxLayer:
            return detail::validate_priorbox_layer<NEPriorBoxLayer>(*polymorphic_downcast<PriorBoxLayerNode *>(node));
        case NodeType::ReorgLayer:
            return detail::validate_reorg_layer<NEReorgLayer>(*polymorphic_downcast<ReorgLayerNode *>(node));
        case NodeType::ReshapeLayer:
            return detail::validate_reshape_layer<NEReshapeLayer>(*polymorphic_downcast<ReshapeLayerNode *>(node));
        case NodeType::ROIAlignLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : ROIAlignLayer");
        case NodeType::SliceLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : SliceLayer");
        case NodeType::UpsampleLayer:
            return detail::validate_upsample_layer<NEUpsampleLayer>(*polymorphic_downcast<UpsampleLayerNode *>(node));
        case NodeType::YOLOLayer:
            return detail::validate_yolo_layer<NEYOLOLayer>(*polymorphic_downcast<YOLOLayerNode *>(node));
        default:
            return Status{};
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
