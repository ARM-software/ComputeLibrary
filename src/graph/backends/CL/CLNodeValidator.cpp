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
#include "arm_compute/graph/backends/CL/CLNodeValidator.h"

#include "arm_compute/graph/backends/ValidateHelpers.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CPP/CPPFunctions.h"
#include "src/core/CL/kernels/CLDepthConvertLayerKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpMatrixMultiplyNativeKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpOffsetContributionKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "src/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "src/core/CL/kernels/CLIm2ColKernel.h"
#include "src/core/CL/kernels/CLQLSTMLayerNormalizationKernel.h"
#include "src/core/CL/kernels/CLWeightsReshapeKernel.h"
#include "support/Cast.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Collection of CL element-wise functions */
struct CLEltwiseLayerFunctions
{
    using ArithmeticAddition      = CLArithmeticAddition;
    using ArithmeticSubtraction   = CLArithmeticSubtraction;
    using PixelWiseMultiplication = CLPixelWiseMultiplication;
    using ElementwiseMax          = CLElementwiseMax;
};

/** Collection of CL unary element-wise functions */
struct CLUnaryEltwiseLayerFunctions
{
    using ExpLayer = CLExpLayer;
};

Status CLNodeValidator::validate(INode *node)
{
    if(node == nullptr)
    {
        return Status{};
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::ArgMinMaxLayer:
            return detail::validate_arg_min_max_layer<CLArgMinMaxLayer>(*polymorphic_downcast<ArgMinMaxLayerNode *>(node));
        case NodeType::BoundingBoxTransformLayer:
            return detail::validate_bounding_box_transform_layer<CLBoundingBoxTransform>(*polymorphic_downcast<BoundingBoxTransformLayerNode *>(node));
        case NodeType::ChannelShuffleLayer:
            return detail::validate_channel_shuffle_layer<CLChannelShuffleLayer>(*polymorphic_downcast<ChannelShuffleLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return detail::validate_convolution_layer<CLConvolutionLayer,
                   CLDirectConvolutionLayer,
                   CLGEMMConvolutionLayer,
                   CLWinogradConvolutionLayer>(*polymorphic_downcast<ConvolutionLayerNode *>(node));
        case NodeType::DepthToSpaceLayer:
            return detail::validate_depth_to_space_layer<CLDepthToSpaceLayer>(*polymorphic_downcast<DepthToSpaceLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::validate_depthwise_convolution_layer<CLDepthwiseConvolutionLayer>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::DequantizationLayer:
            return detail::validate_dequantization_layer<CLDequantizationLayer>(*polymorphic_downcast<DequantizationLayerNode *>(node));
        case NodeType::DetectionOutputLayer:
            return detail::validate_detection_output_layer<CPPDetectionOutputLayer>(*polymorphic_downcast<DetectionOutputLayerNode *>(node));
        case NodeType::DetectionPostProcessLayer:
            return detail::validate_detection_post_process_layer<CPPDetectionPostProcessLayer>(*polymorphic_downcast<DetectionPostProcessLayerNode *>(node));
        case NodeType::GenerateProposalsLayer:
            return detail::validate_generate_proposals_layer<CLGenerateProposalsLayer>(*polymorphic_downcast<GenerateProposalsLayerNode *>(node));
        case NodeType::L2NormalizeLayer:
            return detail::validate_l2_normalize_layer<CLL2NormalizeLayer>(*polymorphic_downcast<L2NormalizeLayerNode *>(node));
        case NodeType::NormalizePlanarYUVLayer:
            return detail::validate_normalize_planar_yuv_layer<CLNormalizePlanarYUVLayer>(*polymorphic_downcast<NormalizePlanarYUVLayerNode *>(node));
        case NodeType::PadLayer:
            return detail::validate_pad_layer<CLPadLayer>(*polymorphic_downcast<PadLayerNode *>(node));
        case NodeType::PermuteLayer:
            return detail::validate_permute_layer<CLPermute>(*polymorphic_downcast<PermuteLayerNode *>(node));
        case NodeType::PReluLayer:
            return detail::validate_prelu_layer<CLPReluLayer>(*polymorphic_downcast<PReluLayerNode *>(node));
        case NodeType::PriorBoxLayer:
            return detail::validate_priorbox_layer<CLPriorBoxLayer>(*polymorphic_downcast<PriorBoxLayerNode *>(node));
        case NodeType::QuantizationLayer:
            return detail::validate_quantization_layer<CLQuantizationLayer>(*polymorphic_downcast<QuantizationLayerNode *>(node));
        case NodeType::ReductionOperationLayer:
            return detail::validate_reduction_operation_layer<CLReductionOperation>(*polymorphic_downcast<ReductionLayerNode *>(node));
        case NodeType::ReorgLayer:
            return detail::validate_reorg_layer<CLReorgLayer>(*polymorphic_downcast<ReorgLayerNode *>(node));
        case NodeType::ReshapeLayer:
            return detail::validate_reshape_layer<CLReshapeLayer>(*polymorphic_downcast<ReshapeLayerNode *>(node));
        case NodeType::ROIAlignLayer:
            return detail::validate_roi_align_layer<CLROIAlignLayer>(*polymorphic_downcast<ROIAlignLayerNode *>(node));
        case NodeType::SliceLayer:
            return detail::validate_slice_layer<CLSlice>(*polymorphic_downcast<SliceLayerNode *>(node));
        case NodeType::StridedSliceLayer:
            return detail::validate_strided_slice_layer<CLStridedSlice>(*polymorphic_downcast<StridedSliceLayerNode *>(node));
        case NodeType::UpsampleLayer:
            return detail::validate_upsample_layer<CLUpsampleLayer>(*polymorphic_downcast<UpsampleLayerNode *>(node));
        case NodeType::YOLOLayer:
            return detail::validate_yolo_layer<CLYOLOLayer>(*polymorphic_downcast<YOLOLayerNode *>(node));
        case NodeType::EltwiseLayer:
            return detail::validate_eltwise_Layer<CLEltwiseLayerFunctions>(*polymorphic_downcast<EltwiseLayerNode *>(node));
        case NodeType::UnaryEltwiseLayer:
            return detail::validate_unary_eltwise_layer<CLUnaryEltwiseLayerFunctions>(*polymorphic_downcast<UnaryEltwiseLayerNode *>(node));
        default:
            return Status{};
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
