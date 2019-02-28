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
#include "arm_compute/graph/backends/GLES/GCFunctionFactory.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/backends/FunctionHelpers.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCFunctions.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Target specific information structure used to pass information to the layer templates */
struct GCTargetInfo
{
    using TensorType = arm_compute::IGCTensor;
    static Target TargetType;
};

Target GCTargetInfo::TargetType = Target::GC;

/** Collection of GC convolution functions */
struct GCConvolutionLayerFunctions
{
    using GenericConvolutionLayer = GCConvolutionLayer;
    using GEMMConvolutionLayer    = GCConvolutionLayer;
    using DirectConvolutionLayer  = GCDirectConvolutionLayer;
};

/** Collection of GC depthwise convolution functions */
struct GCDepthwiseConvolutionLayerFunctions
{
    using DepthwiseConvolutionLayer3x3 = GCDepthwiseConvolutionLayer3x3;
};

/** Collection of GC element-wise functions */
struct GCEltwiseFunctions
{
    using Addition       = GCArithmeticAddition;
    using Multiplication = GCPixelWiseMultiplication;
};

namespace detail
{
// Specialize functions
template <>
std::unique_ptr<IFunction> create_concatenate_layer<GCDepthConcatenateLayer, GCTargetInfo>(ConcatenateLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating Concatenate node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Return nullptr if depth concatenate is switched off
    if(!node.is_enabled())
    {
        return nullptr;
    }

    // Extract IO and info
    std::vector<GCTargetInfo::TensorType *> inputs;
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        inputs.push_back(get_backing_tensor<GCTargetInfo>(node.input(i)));
    }
    typename GCTargetInfo::TensorType *output = get_backing_tensor<GCTargetInfo>(node.output(0));

    // Create and configure function
    auto func = support::cpp14::make_unique<GCDepthConcatenateLayer>();
    func->configure(inputs, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Target " << GCTargetInfo::TargetType
                               << " Data Type: " << output->info()->data_type()
                               << " Shape: " << output->info()->tensor_shape()
                               << " Num Inputs: " << inputs.size()
                               << std::endl);

    return std::move(func);
}

template <>
std::unique_ptr<IFunction> create_convolution_layer<GCConvolutionLayerFunctions, GCTargetInfo>(ConvolutionLayerNode &node, GraphContext &ctx)
{
    validate_node<GCTargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    GCTargetInfo::TensorType *input   = get_backing_tensor<GCTargetInfo>(node.input(0));
    GCTargetInfo::TensorType *weights = get_backing_tensor<GCTargetInfo>(node.input(1));
    GCTargetInfo::TensorType *biases  = get_backing_tensor<GCTargetInfo>(node.input(2));
    GCTargetInfo::TensorType *output  = get_backing_tensor<GCTargetInfo>(node.output(0));

    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo       conv_info      = node.convolution_info();
    const ConvolutionMethod   conv_algorithm = node.convolution_method();
    const ActivationLayerInfo fused_act      = node.fused_activation();

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, GCTargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;

    if(conv_algorithm == ConvolutionMethod::Direct)
    {
        std::tie(func, func_name) = create_named_function<GCConvolutionLayerFunctions::DirectConvolutionLayer>(
                                        std::string("DirectConvolutionLayer"),
                                        input, weights, biases, output, conv_info, fused_act);
    }
    else
    {
        std::tie(func, func_name) = create_named_memory_managed_function<GCConvolutionLayerFunctions::GenericConvolutionLayer>(
                                        std::string("ConvolutionLayer"), mm,
                                        input, weights, biases, output, conv_info, WeightsInfo(), Size2D(1U, 1U), fused_act);
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << func_name
                               << " Data Type: " << input->info()->data_type()
                               << " Input QuantInfo: " << input->info()->quantization_info()
                               << " Weights QuantInfo: " << weights->info()->quantization_info()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << (fused_act.enabled() ? " " + to_string(fused_act.activation()) : "")
                               << std::endl);
    return func;
}

template <>
std::unique_ptr<IFunction> create_depthwise_convolution_layer<GCDepthwiseConvolutionLayerFunctions, GCTargetInfo>(DepthwiseConvolutionLayerNode &node)
{
    validate_node<GCTargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    GCTargetInfo::TensorType *input   = get_backing_tensor<GCTargetInfo>(node.input(0));
    GCTargetInfo::TensorType *weights = get_backing_tensor<GCTargetInfo>(node.input(1));
    GCTargetInfo::TensorType *biases  = get_backing_tensor<GCTargetInfo>(node.input(2));
    GCTargetInfo::TensorType *output  = get_backing_tensor<GCTargetInfo>(node.output(0));

    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo              conv_info        = node.convolution_info();
    const DepthwiseConvolutionMethod dwc_algorithm    = node.depthwise_convolution_method();
    const ActivationLayerInfo        fused_act        = node.fused_activation();
    const int                        depth_multiplier = node.depth_multiplier();

    // Create and configure function (we assume that functions have been validated before creation)
    std::unique_ptr<IFunction> func;
    std::string                func_name;
    if(dwc_algorithm == DepthwiseConvolutionMethod::Optimized3x3)
    {
        std::tie(func, func_name) = create_named_function<GCDepthwiseConvolutionLayerFunctions::DepthwiseConvolutionLayer3x3>(
                                        std::string("DepthwiseConvolutionLayer3x3"),
                                        input, weights, biases, output, conv_info, depth_multiplier, fused_act);
    }
    else
    {
        ARM_COMPUTE_ERROR("Generic DepthwiseConvolutionLayer is not supported in GLES backend");
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << func_name
                               << " Target " << GCTargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input QuantInfo: " << input->info()->quantization_info()
                               << " Weights QuantInfo: " << weights->info()->quantization_info()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Depth multiplier: " << depth_multiplier
                               << (fused_act.enabled() ? " " + to_string(fused_act.activation()) : "")
                               << std::endl);
    return func;
}

template <>
std::unique_ptr<IFunction> create_eltwise_layer<GCEltwiseFunctions, GCTargetInfo>(EltwiseLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(
        "Creating GC EltwiseLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 2);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    GCTargetInfo::TensorType *input1         = get_backing_tensor<GCTargetInfo>(node.input(0));
    GCTargetInfo::TensorType *input2         = get_backing_tensor<GCTargetInfo>(node.input(1));
    GCTargetInfo::TensorType *output         = get_backing_tensor<GCTargetInfo>(node.output(0));
    const EltwiseOperation    eltwise_op     = node.eltwise_operation();
    const ConvertPolicy       convert_policy = node.convert_policy();
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(input2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    std::unique_ptr<IFunction> func = nullptr;
    std::string                func_name;
    if(eltwise_op == EltwiseOperation::Add)
    {
        std::tie(func, func_name) = create_named_function<GCEltwiseFunctions::Addition>(
                                        std::string("GCArithmeticAddition"),
                                        input1, input2, output, convert_policy);
    }
    else if(eltwise_op == EltwiseOperation::Sub)
    {
        ARM_COMPUTE_ERROR("Arithmetic subtraction is not supported in GLES backend");
    }
    else if(eltwise_op == EltwiseOperation::Mul)
    {
        std::tie(func, func_name) = create_named_function<GCEltwiseFunctions::Multiplication>(
                                        std::string("PixelWiseMultiplication"),
                                        input1, input2, output, 1.f);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported element-wise operation!");
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << GCTargetInfo::TargetType
                               << " Operation: " << func_name
                               << " Data Type: " << input1->info()->data_type()
                               << " Shape: " << input1->info()->tensor_shape()
                               << std::endl);

    return func;
}
} //namespace detail

std::unique_ptr<IFunction> GCFunctionFactory::create(INode *node, GraphContext &ctx)
{
    if(node == nullptr)
    {
        return nullptr;
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::ActivationLayer:
            return detail::create_activation_layer<GCActivationLayer, GCTargetInfo>(*polymorphic_downcast<ActivationLayerNode *>(node));
        case NodeType::BatchNormalizationLayer:
            return detail::create_batch_normalization_layer<GCBatchNormalizationLayer, GCTargetInfo>(*polymorphic_downcast<BatchNormalizationLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return detail::create_convolution_layer<GCConvolutionLayerFunctions, GCTargetInfo>(*polymorphic_downcast<ConvolutionLayerNode *>(node), ctx);
        case NodeType::ConcatenateLayer:
            return detail::create_concatenate_layer<GCDepthConcatenateLayer, GCTargetInfo>(*polymorphic_downcast<ConcatenateLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::create_depthwise_convolution_layer<GCDepthwiseConvolutionLayerFunctions, GCTargetInfo>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::EltwiseLayer:
            return detail::create_eltwise_layer<GCEltwiseFunctions, GCTargetInfo>(*polymorphic_downcast<EltwiseLayerNode *>(node));
        case NodeType::FullyConnectedLayer:
            return detail::create_fully_connected_layer<GCFullyConnectedLayer, GCTargetInfo>(*polymorphic_downcast<FullyConnectedLayerNode *>(node), ctx);
        case NodeType::NormalizationLayer:
            return detail::create_normalization_layer<GCNormalizationLayer, GCTargetInfo>(*polymorphic_downcast<NormalizationLayerNode *>(node), ctx);
        case NodeType::NormalizePlanarYUVLayer:
            return detail::create_normalize_planar_yuv_layer<GCNormalizePlanarYUVLayer, GCTargetInfo>(*polymorphic_downcast<NormalizePlanarYUVLayerNode *>(node));
        case NodeType::PoolingLayer:
            return detail::create_pooling_layer<GCPoolingLayer, GCTargetInfo>(*polymorphic_downcast<PoolingLayerNode *>(node));
        case NodeType::ResizeLayer:
            return detail::create_resize_layer<GCScale, GCTargetInfo>(*polymorphic_downcast<ResizeLayerNode *>(node));
        case NodeType::SoftmaxLayer:
            return detail::create_softmax_layer<GCSoftmaxLayer, GCTargetInfo>(*polymorphic_downcast<SoftmaxLayerNode *>(node), ctx);
        default:
            return nullptr;
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
