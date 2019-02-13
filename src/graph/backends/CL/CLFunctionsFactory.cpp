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
#include "arm_compute/graph/backends/CL/CLFunctionFactory.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/backends/FunctionHelpers.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CPP/CPPFunctions.h"

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
// TODO (isagot01): Remove once we support heterogeneous scheduling at function level
/** Wrapper for the CPP Function in the OpenCL backend **/
class CPPWrapperFunction : public IFunction
{
public:
    /* Default constructor */
    CPPWrapperFunction()
        : _tensors(), _func(nullptr)
    {
    }

    void run() override
    {
        for(auto &tensor : _tensors)
        {
            tensor->map(CLScheduler::get().queue());
        }
        _func->run();

        for(auto &tensor : _tensors)
        {
            tensor->unmap(CLScheduler::get().queue());
        }
    }

    void register_tensor(ICLTensor *tensor)
    {
        _tensors.push_back(tensor);
    }

    void register_function(std::unique_ptr<IFunction> function)
    {
        _func = std::move(function);
    }

private:
    std::vector<arm_compute::ICLTensor *> _tensors;
    std::unique_ptr<IFunction>            _func;
};

namespace detail
{
// Specialized functions
template <>
std::unique_ptr<IFunction> create_detection_output_layer<CPPDetectionOutputLayer, CLTargetInfo>(DetectionOutputLayerNode &node)
{
    validate_node<CLTargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    CLTargetInfo::TensorType      *input0      = get_backing_tensor<CLTargetInfo>(node.input(0));
    CLTargetInfo::TensorType      *input1      = get_backing_tensor<CLTargetInfo>(node.input(1));
    CLTargetInfo::TensorType      *input2      = get_backing_tensor<CLTargetInfo>(node.input(2));
    CLTargetInfo::TensorType      *output      = get_backing_tensor<CLTargetInfo>(node.output(0));
    const DetectionOutputLayerInfo detect_info = node.detection_output_info();

    ARM_COMPUTE_ERROR_ON(input0 == nullptr);
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(input2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = support::cpp14::make_unique<CPPDetectionOutputLayer>();
    func->configure(input0, input1, input2, output, detect_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << CLTargetInfo::TargetType
                               << " Data Type: " << input0->info()->data_type()
                               << " Input0 shape: " << input0->info()->tensor_shape()
                               << " Input1 shape: " << input1->info()->tensor_shape()
                               << " Input2 shape: " << input2->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " DetectionOutputLayer info: " << detect_info
                               << std::endl);

    auto wrap_function = support::cpp14::make_unique<CPPWrapperFunction>();
    ;
    wrap_function->register_function(std::move(func));
    wrap_function->register_tensor(input0);
    wrap_function->register_tensor(input1);
    wrap_function->register_tensor(input2);
    wrap_function->register_tensor(output);

    return std::move(wrap_function);
}
} // namespace detail

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
        case NodeType::BoundingBoxTransformLayer:
            return detail::create_bounding_box_transform_layer<CLBoundingBoxTransform, CLTargetInfo>(*polymorphic_downcast<BoundingBoxTransformLayerNode *>(node));
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
        case NodeType::DetectionOutputLayer:
            return detail::create_detection_output_layer<CPPDetectionOutputLayer, CLTargetInfo>(*polymorphic_downcast<DetectionOutputLayerNode *>(node));
        case NodeType::EltwiseLayer:
            return detail::create_eltwise_layer<CLEltwiseFunctions, CLTargetInfo>(*polymorphic_downcast<EltwiseLayerNode *>(node));
        case NodeType::FlattenLayer:
            return detail::create_flatten_layer<CLFlattenLayer, CLTargetInfo>(*polymorphic_downcast<FlattenLayerNode *>(node));
        case NodeType::FullyConnectedLayer:
            return detail::create_fully_connected_layer<CLFullyConnectedLayer, CLTargetInfo>(*polymorphic_downcast<FullyConnectedLayerNode *>(node), ctx);
        case NodeType::GenerateProposalsLayer:
            return detail::create_generate_proposals_layer<CLGenerateProposalsLayer, CLTargetInfo>(*polymorphic_downcast<GenerateProposalsLayerNode *>(node), ctx);
        case NodeType::NormalizationLayer:
            return detail::create_normalization_layer<CLNormalizationLayer, CLTargetInfo>(*polymorphic_downcast<NormalizationLayerNode *>(node), ctx);
        case NodeType::NormalizePlanarYUVLayer:
            return detail::create_normalize_planar_yuv_layer<CLNormalizePlanarYUVLayer, CLTargetInfo>(*polymorphic_downcast<NormalizePlanarYUVLayerNode *>(node));
        case NodeType::PadLayer:
            return detail::create_pad_layer<CLPadLayer, CLTargetInfo>(*polymorphic_downcast<PadLayerNode *>(node));
        case NodeType::PermuteLayer:
            return detail::create_permute_layer<CLPermute, CLTargetInfo>(*polymorphic_downcast<PermuteLayerNode *>(node));
        case NodeType::PoolingLayer:
            return detail::create_pooling_layer<CLPoolingLayer, CLTargetInfo>(*polymorphic_downcast<PoolingLayerNode *>(node));
        case NodeType::PriorBoxLayer:
            return detail::create_priorbox_layer<CLPriorBoxLayer, CLTargetInfo>(*polymorphic_downcast<PriorBoxLayerNode *>(node));
        case NodeType::ReorgLayer:
            return detail::create_reorg_layer<CLReorgLayer, CLTargetInfo>(*polymorphic_downcast<ReorgLayerNode *>(node));
        case NodeType::ReshapeLayer:
            return detail::create_reshape_layer<CLReshapeLayer, CLTargetInfo>(*polymorphic_downcast<ReshapeLayerNode *>(node));
        case NodeType::ResizeLayer:
            return detail::create_resize_layer<CLScale, CLTargetInfo>(*polymorphic_downcast<ResizeLayerNode *>(node));
        case NodeType::ROIAlignLayer:
            return detail::create_roi_align_layer<CLROIAlignLayer, CLTargetInfo>(*polymorphic_downcast<ROIAlignLayerNode *>(node));
        case NodeType::SliceLayer:
            return detail::create_slice_layer<CLSlice, CLTargetInfo>(*polymorphic_downcast<SliceLayerNode *>(node));
        case NodeType::SoftmaxLayer:
            return detail::create_softmax_layer<CLSoftmaxLayer, CLTargetInfo>(*polymorphic_downcast<SoftmaxLayerNode *>(node), ctx);
        case NodeType::UpsampleLayer:
            return detail::create_upsample_layer<CLUpsampleLayer, CLTargetInfo>(*polymorphic_downcast<UpsampleLayerNode *>(node), ctx);
        case NodeType::YOLOLayer:
            return detail::create_yolo_layer<CLYOLOLayer, CLTargetInfo>(*polymorphic_downcast<YOLOLayerNode *>(node), ctx);
        default:
            return nullptr;
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
