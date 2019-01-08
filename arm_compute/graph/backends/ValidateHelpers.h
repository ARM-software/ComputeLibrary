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
#ifndef __ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_VALIDATE_HELPERS_H__
#define __ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_VALIDATE_HELPERS_H__

#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorInfo.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
namespace detail
{
/** Returns backing tensor info of a given tensor
 *
 * @param[in] tensor Tensor to extract the backing tensor from
 *
 * @return Backing tensor tensor info if present else nullptr
 */
inline arm_compute::ITensorInfo *get_backing_tensor_info(arm_compute::graph::Tensor *tensor)
{
    return ((tensor == nullptr) || (tensor->handle() == nullptr)) ? nullptr : tensor->handle()->tensor().info();
}

/** Validates a Bounding Box Transform layer node
 *
 * @tparam BoundingBoxTransformLayer  Bounding Box Transform layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename BoundingBoxTransformLayer>
Status validate_bounding_box_transform_layer(BoundingBoxTransformLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating BoundingBoxTransformLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo      *input     = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo      *deltas    = get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo      *output    = get_backing_tensor_info(node.output(0));
    const BoundingBoxTransformInfo bbox_info = node.info();

    return BoundingBoxTransformLayer::validate(input, output, deltas, bbox_info);
}

/** Validates a Channel Shuffle layer node
 *
 * @tparam ChannelShuffleLayer  Channel Shuffle layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ChannelShuffleLayer>
Status validate_channel_shuffle_layer(ChannelShuffleLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating ChannelShuffle node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input      = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output     = get_backing_tensor_info(node.output(0));
    const unsigned int        num_groups = node.num_groups();

    return ChannelShuffleLayer::validate(input, output, num_groups);
}

/** Validates a Convolution layer node
 *
 * @tparam ConvolutionLayer          Default Convolution layer function type
 * @tparam DirectConvolutionLayer    Direct Convolution layer function type
 * @tparam GEMMConvolutionLayer      GEMM Convolution layer function type
 * @tparam WinogradConvolutionLayer  Winograd Convolution layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ConvolutionLayer, typename DirectConvolutionLayer, typename GEMMConvolutionLayer, typename WinogradConvolutionLayer>
Status validate_convolution_layer(ConvolutionLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating ConvolutionLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input   = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *weights = get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo *biases  = get_backing_tensor_info(node.input(2));
    arm_compute::ITensorInfo *output  = get_backing_tensor_info(node.output(0));

    if(is_data_type_quantized_asymmetric(input->data_type()))
    {
        biases->set_data_type(DataType::S32);
    }

    const PadStrideInfo     conv_info      = node.convolution_info();
    const ConvolutionMethod conv_algorithm = node.convolution_method();
    const bool              fast_math      = node.fast_math_hint() == FastMathHint::Enabled;
    const unsigned int      num_groups     = node.num_groups();

    // Validate function
    Status status{};
    switch(conv_algorithm)
    {
        case ConvolutionMethod::Direct:
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups != 1, "DirectConvolutionLayer does not support grouping!");
            status = DirectConvolutionLayer::validate(input, weights, biases, output, conv_info);
            break;
        case ConvolutionMethod::GEMM:
            status = GEMMConvolutionLayer::validate(input, weights, biases, output, conv_info,
                                                    WeightsInfo(), Size2D(1, 1), ActivationLayerInfo(), num_groups);
            break;
        case ConvolutionMethod::Winograd:
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups != 1, "WinogradConvolutionLayer does not support grouping!");
            status = WinogradConvolutionLayer::validate(input, weights, biases, output, conv_info, ActivationLayerInfo(), fast_math);
            break;
        case ConvolutionMethod::Default:
            status = ConvolutionLayer::validate(input, weights, biases, output, conv_info,
                                                WeightsInfo(), Size2D(1, 1), ActivationLayerInfo(), fast_math, num_groups);
            break;
        default:
            ARM_COMPUTE_RETURN_ERROR_MSG("Unsupported convolution method");
    }

    return status;
}

/** Validates a Depthwise Convolution layer node
 *
 * @tparam DepthwiseConvolutionLayer    Default Depthwise Convolution layer type
 * @tparam DepthwiseConvolutionLayer3x3 Optimized 3x3 Depthwise Convolution layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DepthwiseConvolutionLayer, typename DepthwiseConvolutionLayer3x3>
Status validate_depthwise_convolution_layer(DepthwiseConvolutionLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating DepthwiseConvolutionLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input   = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *weights = detail::get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo *biases  = get_backing_tensor_info(node.input(2));
    arm_compute::ITensorInfo *output  = get_backing_tensor_info(node.output(0));

    const PadStrideInfo              conv_info        = node.convolution_info();
    const DepthwiseConvolutionMethod dwc_algorithm    = node.depthwise_convolution_method();
    const int                        depth_multiplier = node.depth_multiplier();

    // Validate function
    Status status{};
    switch(dwc_algorithm)
    {
        case DepthwiseConvolutionMethod::Default:
        case DepthwiseConvolutionMethod::GEMV:
            status = DepthwiseConvolutionLayer::validate(input, weights, biases, output, conv_info, depth_multiplier);
            break;
        case DepthwiseConvolutionMethod::Optimized3x3:
            status = DepthwiseConvolutionLayer3x3::validate(input, weights, biases, output, conv_info, depth_multiplier);
            break;
        default:
            ARM_COMPUTE_RETURN_ERROR_MSG("Unsupported depthwise convolution method");
    }

    return status;
}

/** Validates a detection output layer node
 *
 * @tparam DetectionOutputLayer DetectionOutput layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DetectionOutputLayer>
Status validate_detection_output_layer(DetectionOutputLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating DetectionOutputLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo      *input0      = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo      *input1      = get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo      *input2      = get_backing_tensor_info(node.input(2));
    arm_compute::ITensorInfo      *output      = get_backing_tensor_info(node.output(0));
    const DetectionOutputLayerInfo detect_info = node.detection_output_info();

    return DetectionOutputLayer::validate(input0, input1, input2, output, detect_info);
}
/** Validates a detection post process layer node
 *
 * @tparam DetectionPostProcessLayer DetectionOutput layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DetectionPostProcessLayer>
Status validate_detection_post_process_layer(DetectionPostProcessLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating DetectionPostProcessLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 4);

    // Extract IO and info
    arm_compute::ITensorInfo           *input0      = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo           *input1      = get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo           *input2      = get_backing_tensor_info(node.input(2));
    arm_compute::ITensorInfo           *output0     = get_backing_tensor_info(node.output(0));
    arm_compute::ITensorInfo           *output1     = get_backing_tensor_info(node.output(1));
    arm_compute::ITensorInfo           *output2     = get_backing_tensor_info(node.output(2));
    arm_compute::ITensorInfo           *output3     = get_backing_tensor_info(node.output(3));
    const DetectionPostProcessLayerInfo detect_info = node.detection_post_process_info();

    return DetectionPostProcessLayer::validate(input0, input1, input2, output0, output1, output2, output3, detect_info);
}

/** Validates a Generate Proposals layer node
 *
 * @tparam GenerateProposalsLayer Generate Proposals layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename GenerateProposalsLayer>
Status validate_generate_proposals_layer(GenerateProposalsLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating GenerateProposalsLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 3);

    // Extract IO and info
    arm_compute::ITensorInfo   *scores              = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo   *deltas              = detail::get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo   *anchors             = detail::get_backing_tensor_info(node.input(2));
    arm_compute::ITensorInfo   *proposals           = get_backing_tensor_info(node.output(0));
    arm_compute::ITensorInfo   *scores_out          = get_backing_tensor_info(node.output(1));
    arm_compute::ITensorInfo   *num_valid_proposals = get_backing_tensor_info(node.output(2));
    const GenerateProposalsInfo info                = node.info();

    return GenerateProposalsLayer::validate(scores, deltas, anchors, proposals, scores_out, num_valid_proposals, info);
}

/** Validates a NormalizePlanarYUV layer node
 *
 * @tparam NormalizePlanarYUVLayer layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename NormalizePlanarYUVLayer>
Status validate_normalize_planar_yuv_layer(NormalizePlanarYUVLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating NormalizePlanarYUVLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *mean   = detail::get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo *std    = detail::get_backing_tensor_info(node.input(2));
    arm_compute::ITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return NormalizePlanarYUVLayer::validate(input, output, mean, std);
}

/** Validates a pad layer node
 *
 * @tparam PadLayer Pad layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename PadLayer>
Status validate_pad_layer(PadLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating PadLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input   = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output  = get_backing_tensor_info(node.output(0));
    const PaddingList        &padding = node.padding();

    return PadLayer::validate(input, output, padding);
}

/** Validates a permute layer node
 *
 * @tparam PermuteLayer Permute layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename PermuteLayer>
Status validate_permute_layer(PermuteLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating PermuteLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input  = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output = get_backing_tensor_info(node.output(0));
    const PermutationVector &perm   = node.permutation_vector();

    return PermuteLayer::validate(input, output, perm);
}
/** Validates a priorbox layer node
 *
 * @tparam PriorBoxLayer PriorBox layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename PriorBoxLayer>
Status validate_priorbox_layer(PriorBoxLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating PriorBoxLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input0     = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *input1     = get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo *output     = get_backing_tensor_info(node.output(0));
    const PriorBoxLayerInfo   prior_info = node.priorbox_info();

    return PriorBoxLayer::validate(input0, input1, output, prior_info);
}

/** Validates a Quantization layer node
 *
 * @tparam QuantizationLayer Quantization layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename QuantizationLayer>
Status validate_quantization_layer(QuantizationLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating QuantizationLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    arm_compute::ITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return QuantizationLayer::validate(input, output);
}

/** Validates a Reorg layer node
 *
 * @tparam ReorgLayer Reorg layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ReorgLayer>
Status validate_reorg_layer(ReorgLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating ReorgLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    arm_compute::ITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return ReorgLayer::validate(input, output, node.stride());
}

/** Validates a Reshape layer node
 *
 * @tparam ReshapeLayer Reshape layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ReshapeLayer>
Status validate_reshape_layer(ReshapeLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating ReshapeLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    arm_compute::ITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output = detail::get_backing_tensor_info(node.output(0));

    // Validate function
    return ReshapeLayer::validate(input, output);
}

/** Validates a ROI Align layer node
 *
 * @tparam ROIAlignLayer ROIAlign layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ROIAlignLayer>
Status validate_roi_align_layer(ROIAlignLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating ROIAlignLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    arm_compute::ITensorInfo *input     = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *rois      = detail::get_backing_tensor_info(node.input(1));
    arm_compute::ITensorInfo *output    = detail::get_backing_tensor_info(node.output(0));
    const ROIPoolingLayerInfo &pool_info = node.pooling_info();

    // Validate function
    return ROIAlignLayer::validate(input, rois, output, pool_info);
}

/** Validates a Slice layer node
 *
 * @tparam SliceLayer Slice layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename SliceLayer>
Status validate_slice_layer(SliceLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating Slice node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    arm_compute::ITensorInfo *input  = get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output = get_backing_tensor_info(node.output(0));
    const Coordinates         starts = node.starts();
    const Coordinates         ends   = node.ends();

    return SliceLayer::validate(input, output, starts, ends);
}

/** Validates a Upsample layer node
 *
 * @tparam UpsampleLayer Upsample layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename UpsampleLayer>
Status validate_upsample_layer(UpsampleLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating UpsampleLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    arm_compute::ITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return UpsampleLayer::validate(input, output, node.info(), node.upsampling_policy());
}
/** Validates a YOLO layer node
 *
 * @tparam YOLOLayer YOLO layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename YOLOLayer>
Status validate_yolo_layer(YOLOLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating YOLOLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    arm_compute::ITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    arm_compute::ITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return YOLOLayer::validate(input, output, node.activation_info(), node.num_classes());
}
} // namespace detail
} // namespace backends
} // namespace graph
} // namespace arm_compute

#endif /* __ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_VALIDATE_HELPERS_H__ */
