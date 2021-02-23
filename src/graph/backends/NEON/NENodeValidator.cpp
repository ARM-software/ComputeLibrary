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
#include "arm_compute/graph/backends/NEON/NENodeValidator.h"

#include "arm_compute/graph/backends/ValidateHelpers.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/runtime/CPP/CPPFunctions.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "src/core/NEON/kernels/NEConvertFullyConnectedWeightsKernel.h"
#include "src/core/NEON/kernels/NEConvertQuantizedSignednessKernel.h"
#include "src/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "src/core/NEON/kernels/NEQLSTMLayerNormalizationKernel.h"
#include "src/core/NEON/kernels/NEWeightsReshapeKernel.h"
#include "support/Cast.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Collection of Neon element-wise functions */
struct NEEltwiseLayerFunctions
{
    using ArithmeticAddition      = NEArithmeticAddition;
    using ArithmeticSubtraction   = NEArithmeticSubtraction;
    using PixelWiseMultiplication = NEPixelWiseMultiplication;
    using ElementwiseMax          = NEElementwiseMax;
};

/** Collection of Neon unary element-wise functions */
struct NEUnaryEltwiseLayerFunctions
{
    using ExpLayer = NEExpLayer;
};

Status NENodeValidator::validate(INode *node)
{
    if(node == nullptr)
    {
        return Status{};
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::ArgMinMaxLayer:
            return detail::validate_arg_min_max_layer<NEArgMinMaxLayer>(*polymorphic_downcast<ArgMinMaxLayerNode *>(node));
        case NodeType::BoundingBoxTransformLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : BoundingBoxTransformLayer");
        case NodeType::ChannelShuffleLayer:
            return detail::validate_channel_shuffle_layer<NEChannelShuffleLayer>(*polymorphic_downcast<ChannelShuffleLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return detail::validate_convolution_layer<NEConvolutionLayer,
                   NEDirectConvolutionLayer,
                   NEGEMMConvolutionLayer,
                   NEWinogradConvolutionLayer>(*polymorphic_downcast<ConvolutionLayerNode *>(node));
        case NodeType::DepthToSpaceLayer:
            return detail::validate_depth_to_space_layer<NEDepthToSpaceLayer>(*polymorphic_downcast<DepthToSpaceLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::validate_depthwise_convolution_layer<NEDepthwiseConvolutionLayer>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::DequantizationLayer:
            return detail::validate_dequantization_layer<NEDequantizationLayer>(*polymorphic_downcast<DequantizationLayerNode *>(node));
        case NodeType::DetectionOutputLayer:
            return detail::validate_detection_output_layer<CPPDetectionOutputLayer>(*polymorphic_downcast<DetectionOutputLayerNode *>(node));
        case NodeType::DetectionPostProcessLayer:
            return detail::validate_detection_post_process_layer<NEDetectionPostProcessLayer>(*polymorphic_downcast<DetectionPostProcessLayerNode *>(node));
        case NodeType::GenerateProposalsLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : GenerateProposalsLayer");
        case NodeType::L2NormalizeLayer:
            return detail::validate_l2_normalize_layer<NEL2NormalizeLayer>(*polymorphic_downcast<L2NormalizeLayerNode *>(node));
        case NodeType::NormalizePlanarYUVLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : NormalizePlanarYUVLayer");
        case NodeType::PadLayer:
            return detail::validate_pad_layer<NEPadLayer>(*polymorphic_downcast<PadLayerNode *>(node));
        case NodeType::PermuteLayer:
            return detail::validate_permute_layer<NEPermute>(*polymorphic_downcast<PermuteLayerNode *>(node));
        case NodeType::PReluLayer:
            return detail::validate_prelu_layer<NEPReluLayer>(*polymorphic_downcast<PReluLayerNode *>(node));
        case NodeType::PriorBoxLayer:
            return detail::validate_priorbox_layer<NEPriorBoxLayer>(*polymorphic_downcast<PriorBoxLayerNode *>(node));
        case NodeType::QuantizationLayer:
            return detail::validate_quantization_layer<NEQuantizationLayer>(*polymorphic_downcast<QuantizationLayerNode *>(node));
        case NodeType::ReductionOperationLayer:
            return detail::validate_reduction_operation_layer<NEReductionOperation>(*polymorphic_downcast<ReductionLayerNode *>(node));
        case NodeType::ReorgLayer:
            return detail::validate_reorg_layer<NEReorgLayer>(*polymorphic_downcast<ReorgLayerNode *>(node));
        case NodeType::ReshapeLayer:
            return detail::validate_reshape_layer<NEReshapeLayer>(*polymorphic_downcast<ReshapeLayerNode *>(node));
        case NodeType::ROIAlignLayer:
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, "Unsupported operation : ROIAlignLayer");
        case NodeType::SliceLayer:
            return detail::validate_slice_layer<NESlice>(*polymorphic_downcast<SliceLayerNode *>(node));
        case NodeType::StridedSliceLayer:
            return detail::validate_strided_slice_layer<NEStridedSlice>(*polymorphic_downcast<StridedSliceLayerNode *>(node));
        case NodeType::EltwiseLayer:
            return detail::validate_eltwise_Layer<NEEltwiseLayerFunctions>(*polymorphic_downcast<EltwiseLayerNode *>(node));
        case NodeType::UnaryEltwiseLayer:
            return detail::validate_unary_eltwise_layer<NEUnaryEltwiseLayerFunctions>(*polymorphic_downcast<UnaryEltwiseLayerNode *>(node));
        default:
            return Status{};
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
