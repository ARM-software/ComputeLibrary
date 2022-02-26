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
#ifndef ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_FUNCTION_HELPERS_H
#define ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_FUNCTION_HELPERS_H

#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/core/experimental/PostOps.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/FusedConvolutionBatchNormalizationFunction.h"
#include "arm_compute/graph/backends/FusedConvolutionBatchNormalizationWithPostOpsFunction.h"
#include "arm_compute/graph/backends/FusedDepthwiseConvolutionBatchNormalizationFunction.h"
#include "arm_compute/graph/backends/Utils.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorInfo.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
namespace detail
{
/** Returns backing tensor of a given tensor
 *
 * @tparam TargetInfo Target information
 *
 * @param[in] tensor Tensor to extract the backing tensor from
 *
 * @return Backing tensor if present else nullptr
 */
template <typename TargetInfo>
typename TargetInfo::TensorType *get_backing_tensor(arm_compute::graph::Tensor *tensor)
{
    typename TargetInfo::TensorType *backing_tensor = nullptr;
    if(tensor != nullptr)
    {
        ARM_COMPUTE_ERROR_ON(tensor->desc().target != TargetInfo::TargetType);
        // Get backing tensor handle
        ITensorHandle *tensor_handle = tensor->handle();
        // Get backing tensor
        backing_tensor = (tensor_handle != nullptr) ? arm_compute::utils::cast::polymorphic_cast<typename TargetInfo::TensorType *>(&tensor_handle->tensor()) : nullptr;
    }

    return backing_tensor;
}

template <typename TargetInfo>
void validate_node(const INode &node, size_t num_expected_inputs, size_t num_expected_outputs)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating " << node.type()
                                  << " Target: " << TargetInfo::TargetType
                                  << " ID: " << node.id()
                                  << node.name()
                                  << std::endl);

    ARM_COMPUTE_ERROR_ON(TargetInfo::TargetType != node.assigned_target());
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != num_expected_inputs);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != num_expected_outputs);
    ARM_COMPUTE_UNUSED(node, num_expected_inputs, num_expected_outputs);
}

/** Creates a backend activation layer function
 *
 * @tparam ActivationLayerFunction Backend activation function
 * @tparam TargetInfo              Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend activation layer function
 */
template <typename ActivationLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_activation_layer(ActivationLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input    = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output   = get_backing_tensor<TargetInfo>(node.output(0));
    const ActivationLayerInfo        act_info = node.activation_info();

    // Create function
    auto func = std::make_unique<ActivationLayerFunction>();
    func->configure(input, output, act_info);

    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " Activation function: " << act_info.activation()
                               << " a: " << act_info.a()
                               << " b: " << act_info.b()
                               << " InPlace : " << is_in_place_operation(input, output)
                               << std::endl);

    return std::move(func);
}

/** Creates a backend argminmax layer function
 *
 * @tparam ArgMinMaxLayerFunction Backend activation function
 * @tparam TargetInfo             Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend argminmax layer function
 */
template <typename ArgMinMaxLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_arg_min_max_layer(ArgMinMaxLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    const ReductionOperation         op     = node.reduction_operation();
    unsigned int                     axis   = node.axis();

    // Create function
    auto func = std::make_unique<ArgMinMaxLayerFunction>();
    func->configure(input, axis, output, op);

    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " Reduction Operation: " << op
                               << " axis: " << axis
                               << std::endl);

    return std::move(func);
}

/** Create a backend batch normalization layer function
 *
 * @tparam BatchNormalizationLayerFunction Backend batch normalization function
 * @tparam TargetInfo                      Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend batch normalization layer function
 */
template <typename BatchNormalizationLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_batch_normalization_layer(BatchNormalizationLayerNode &node)
{
    validate_node<TargetInfo>(node, 5 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *mean  = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *var   = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *beta  = get_backing_tensor<TargetInfo>(node.input(3));
    typename TargetInfo::TensorType *gamma = get_backing_tensor<TargetInfo>(node.input(4));

    typename TargetInfo::TensorType *output    = get_backing_tensor<TargetInfo>(node.output(0));
    const float                      epsilon   = node.epsilon();
    const ActivationLayerInfo        fused_act = node.fused_activation();

    // Create and configure function
    auto func = std::make_unique<BatchNormalizationLayerFunction>();
    func->configure(input, output, mean, var, beta, gamma, epsilon, fused_act);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " Epsilon: " << epsilon << " "
                               << (fused_act.enabled() ? to_string(fused_act.activation()) : "")
                               << " InPlace: " << is_in_place_operation(input, output)
                               << std::endl);

    return std::move(func);
}

/** Create a backend batch normalization layer function
 *
 * @tparam BatchNormalizationLayerFunction Backend batch normalization function
 * @tparam TargetInfo                      Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend batch normalization layer function
 */
template <typename FusedLayerTypes, typename TargetInfo>
std::unique_ptr<IFunction> create_fused_convolution_batch_normalization_layer(FusedConvolutionBatchNormalizationNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 7 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *mean    = get_backing_tensor<TargetInfo>(node.input(3));
    typename TargetInfo::TensorType *var     = get_backing_tensor<TargetInfo>(node.input(4));
    typename TargetInfo::TensorType *beta    = get_backing_tensor<TargetInfo>(node.input(5));
    typename TargetInfo::TensorType *gamma   = get_backing_tensor<TargetInfo>(node.input(6));

    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));

    const PadStrideInfo       conv_info  = node.convolution_info();
    const unsigned int        num_groups = node.num_groups();
    const bool                fast_math  = node.fast_math_hint() == FastMathHint::Enabled;
    const ActivationLayerInfo fused_act  = node.fused_activation();
    const float               epsilon    = node.epsilon();

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;

    using FType = FusedConvolutionBatchNormalizationFunction<TargetInfo, FusedLayerTypes>;

    // Create and configure function
    std::tie(func, func_name) = create_named_memory_managed_function<FType>(
                                    std::string("FusedConvolutionBatchNormalizationLayer"), mm, input, weights, biases, output, mean, var, beta, gamma, epsilon, conv_info, num_groups, fast_math, fused_act);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << (fused_act.enabled() ? " " + to_string(fused_act.activation()) : "")
                               << std::endl);
    return std::move(func);
}

/** Create a backend fused depthwise convolution batch normalization layer function
 *
 * @tparam FusedLayerTypes             Fused layer types
 * @tparam TargetInfo                  Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend fused depthwise convolution batch normalization layer function
 */
template <typename FusedLayerTypes, typename TargetInfo>
std::unique_ptr<IFunction> create_fused_depthwise_convolution_batch_normalization_layer(FusedDepthwiseConvolutionBatchNormalizationNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 7 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *mean    = get_backing_tensor<TargetInfo>(node.input(3));
    typename TargetInfo::TensorType *var     = get_backing_tensor<TargetInfo>(node.input(4));
    typename TargetInfo::TensorType *beta    = get_backing_tensor<TargetInfo>(node.input(5));
    typename TargetInfo::TensorType *gamma   = get_backing_tensor<TargetInfo>(node.input(6));

    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));

    const PadStrideInfo       conv_info        = node.convolution_info();
    const unsigned int        depth_multiplier = node.depth_multiplier();
    const ActivationLayerInfo fused_act        = node.fused_activation();
    const float               epsilon          = node.epsilon();

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;

    using FType = FusedDepthwiseConvolutionBatchNormalizationFunction<TargetInfo, FusedLayerTypes>;

    // Create and configure function
    std::tie(func, func_name) = create_named_memory_managed_function<FType>(
                                    std::string("FusedDepthwiseConvolutionBatchNormalizationLayer"), mm, input, weights, biases, output, mean, var, beta, gamma, epsilon, conv_info, depth_multiplier, fused_act);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << (fused_act.enabled() ? " " + to_string(fused_act.activation()) : "")
                               << std::endl);
    return std::move(func);
}

/** Create a backend bounding box transform layer function
 *
 * @tparam BoundingBoxTransformLayerFunction    Backend bounding box transform function
 * @tparam TargetInfo                           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend bounding box transform layer function
 */
template <typename BoundingBoxTransformLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_bounding_box_transform_layer(BoundingBoxTransformLayerNode &node)
{
    validate_node<TargetInfo>(node, 2 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input     = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *deltas    = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *output    = get_backing_tensor<TargetInfo>(node.output(0));
    const BoundingBoxTransformInfo   bbox_info = node.info();

    // Create and configure function
    auto func = std::make_unique<BoundingBoxTransformLayerFunction>();
    func->configure(input, output, deltas, bbox_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " BoundingBox Info img W: " << bbox_info.img_width() << " "
                               << " BoundingBox Info img H: " << bbox_info.img_height() << " "
                               << std::endl);

    return std::move(func);
}

/** Create a backend channel shuffle layer function
 *
 * @tparam ChannelShuffleLayerFunction Backend channel shuffle function
 * @tparam TargetInfo                  Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend channel shuffle layer function
 */
template <typename ChannelShuffleLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_channel_shuffle_layer(ChannelShuffleLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input      = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output     = get_backing_tensor<TargetInfo>(node.output(0));
    const unsigned int               num_groups = node.num_groups();

    // Create function
    auto func = std::make_unique<ChannelShuffleLayerFunction>();
    func->configure(input, output, num_groups);

    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " Num groups: " << num_groups
                               << std::endl);

    return std::move(func);
}

/** Create a backend layer concatenate function
 *
 * @tparam ConcatenateLayerFunction Backend concatenate function
 * @tparam TargetInfo               Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend concatenate layer function
 */
template <typename ConcatenateLayerFunction, typename TargetInfo>
std::unique_ptr<arm_compute::IFunction> create_concatenate_layer(ConcatenateLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating Concatenate node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Return nullptr if depth concatenate is switched off
    if(!node.is_enabled())
    {
        return nullptr;
    }

    // Extract IO and info
    std::vector<typename TargetInfo::SrcTensorType *> inputs;
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        inputs.push_back(get_backing_tensor<TargetInfo>(node.input(i)));
    }
    typename TargetInfo::TensorType *output      = get_backing_tensor<TargetInfo>(node.output(0));
    const DataLayout                 data_layout = node.output(0) != nullptr ? node.output(0)->desc().layout : DataLayout::UNKNOWN;
    const size_t                     concat_axis = get_dimension_idx(data_layout, node.concatenation_axis());

    // Create and configure function
    auto func = std::make_unique<ConcatenateLayerFunction>();
    func->configure(inputs, output, concat_axis);

    // Log info
    const bool         is_quantized = is_data_type_quantized_asymmetric(output->info()->data_type());
    std::ostringstream qss;
    if(is_quantized)
    {
        qss << " Output QuantInfo: " << output->info()->quantization_info();
    }
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << output->info()->data_type()
                               << " Shape: " << output->info()->tensor_shape()
                               << " Num Inputs: " << inputs.size()
                               << " Axis: " << concat_axis
                               << qss.str()
                               << std::endl);

    return std::move(func);
}

/** Create a backend convolution layer function
 *
 * @tparam ConvolutionLayerFunctions Backend convolution functions
 * @tparam TargetInfo                Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend convolution layer function
 */
template <typename ConvolutionLayerFunctions, typename TargetInfo>
std::unique_ptr<IFunction> create_convolution_layer(ConvolutionLayerNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));

    const bool is_quantized = is_data_type_quantized_asymmetric(input->info()->data_type());

    if(is_quantized)
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo       conv_info      = node.convolution_info();
    const unsigned int        num_groups     = node.num_groups();
    const ConvolutionMethod   conv_algorithm = node.convolution_method();
    const bool                fast_math      = node.fast_math_hint() == FastMathHint::Enabled;
    const ActivationLayerInfo fused_act      = node.fused_activation();

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;

    if(conv_algorithm == ConvolutionMethod::Winograd)
    {
        ARM_COMPUTE_ERROR_ON_MSG(num_groups != 1, "WinogradConvolutionLayer does not support grouping!");
        std::tie(func, func_name) = create_named_memory_managed_function<typename ConvolutionLayerFunctions::WinogradConvolutionLayer>(
                                        std::string("WinogradConvolutionLayer"), mm,
                                        input, weights, biases, output, conv_info, fused_act, fast_math);
    }
    else if(conv_algorithm == ConvolutionMethod::Direct)
    {
        ARM_COMPUTE_ERROR_ON_MSG(num_groups != 1, "DirectConvolutionLayer does not support grouping!");
        std::tie(func, func_name) = create_named_function<typename ConvolutionLayerFunctions::DirectConvolutionLayer>(
                                        std::string("DirectConvolutionLayer"),
                                        input, weights, biases, output, conv_info, fused_act);
    }
    else if(conv_algorithm == ConvolutionMethod::GEMM)
    {
        std::tie(func, func_name) = create_named_memory_managed_function<typename ConvolutionLayerFunctions::GEMMConvolutionLayer>(
                                        std::string("GEMMConvolutionLayer"), mm,
                                        input, weights, biases, output, conv_info,
                                        WeightsInfo(), Size2D(1U, 1U), fused_act, num_groups);
    }
    else
    {
        std::tie(func, func_name) = create_named_memory_managed_function<typename ConvolutionLayerFunctions::GenericConvolutionLayer>(
                                        std::string("GenericConvolutionLayer"), mm,
                                        input, weights, biases, output, conv_info,
                                        WeightsInfo(), Size2D(1U, 1U), fused_act, fast_math, num_groups);
    }

    // Log info
    std::ostringstream qss;
    if(is_quantized)
    {
        qss << " Input QuantInfo: " << input->info()->quantization_info()
            << " Weights QuantInfo: " << weights->info()->quantization_info()
            << " Output QuantInfo: " << output->info()->quantization_info();
    }
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << func_name
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Groups: " << num_groups
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << qss.str()
                               << (fused_act.enabled() ? " " + to_string(fused_act.activation()) : "")
                               << std::endl);
    return std::move(func);
}

/** Create a backend convolution layer function with post operator
 *
 * @tparam ConvolutionLayerFunctions Backend convolution functions
 * @tparam TargetInfo                Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend convolution layer function
 */
template <typename ConvolutionLayerFunctions, typename TargetInfo>
std::unique_ptr<IFunction> create_fused_convolution_with_post_op(FusedConvolutionWithPostOpNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 4 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));

    const bool is_quantized = is_data_type_quantized_asymmetric(input->info()->data_type());

    if(is_quantized)
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo       conv_info  = node.convolution_info();
    const unsigned int        num_groups = node.num_groups();
    const ActivationLayerInfo fused_act  = node.fused_activation();

    experimental::PostOpList<typename TargetInfo::TensorType *> post_ops;

    auto &post_op_info_list = node.post_op_info_list();
    for(const auto &post_op_info : post_op_info_list)
    {
        switch(post_op_info->type())
        {
            case PostOpType::Activation:
            {
                const auto act_info = utils::cast::polymorphic_downcast<const ConvPostOpInfoActivation *>(post_op_info.get());
                post_ops.template push_back_op<experimental::PostOpAct<typename TargetInfo::TensorType *>>(act_info->_act);
                break;
            }
            case PostOpType::Eltwise_Add:
            {
                typename TargetInfo::TensorType *add_input    = get_backing_tensor<TargetInfo>(node.input(3));
                const auto                       eltwise_info = utils::cast::polymorphic_downcast<const ConvPostOpInfoEltwiseAdd *>(post_op_info.get());
                post_ops.template push_back_op<experimental::PostOpEltwiseAdd<typename TargetInfo::TensorType *>>(add_input, eltwise_info->_prev_op_dst_pos, eltwise_info->_policy);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported PostOpType");
            }
        }
    }

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;

    // Fuse convolution with post ops is only supported for conv1x1, which is only implemented as gemmconv2d
    std::tie(func, func_name) = create_named_memory_managed_function<typename ConvolutionLayerFunctions::GEMMConvolutionLayer>(
                                    std::string("GEMMConvolutionLayer"), mm,
                                    input, weights, biases, output, conv_info,
                                    WeightsInfo(), Size2D(1U, 1U), fused_act, num_groups, post_ops);

    // Log info
    std::ostringstream qss;
    if(is_quantized)
    {
        qss << " Input QuantInfo: " << input->info()->quantization_info()
            << " Weights QuantInfo: " << weights->info()->quantization_info()
            << " Output QuantInfo: " << output->info()->quantization_info();
    }
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << func_name
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Groups: " << num_groups
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << qss.str()
                               << (fused_act.enabled() ? " " + to_string(fused_act.activation()) : "")
                               << " Post ops" << post_ops
                               << std::endl);
    return std::move(func);
}

/** Create a backend convolution batch normalization layer function with post operator
 *
 * @tparam FusedLayerTypes           Backend convolution functions
 * @tparam TargetInfo                Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend fused convolution with batch normalization layer function
 */
template <typename FusedLayerTypes, typename TargetInfo>
std::unique_ptr<IFunction> create_fused_convolution_batch_normalization_with_post_op(FusedConvolutionBatchNormalizationWithPostOpsNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 8 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *mean    = get_backing_tensor<TargetInfo>(node.input(3));
    typename TargetInfo::TensorType *var     = get_backing_tensor<TargetInfo>(node.input(4));
    typename TargetInfo::TensorType *beta    = get_backing_tensor<TargetInfo>(node.input(5));
    typename TargetInfo::TensorType *gamma   = get_backing_tensor<TargetInfo>(node.input(6));

    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));

    const PadStrideInfo conv_info  = node.convolution_info();
    const unsigned int  num_groups = node.num_groups();
    const bool          fast_math  = node.fast_math_hint() == FastMathHint::Enabled;
    const float         epsilon    = node.epsilon();

    experimental::PostOpList<typename TargetInfo::TensorType *> post_ops;

    auto &post_op_info_list = node.post_op_info_list();
    for(const auto &post_op_info : post_op_info_list)
    {
        switch(post_op_info->type())
        {
            case PostOpType::Activation:
            {
                const auto act_info = utils::cast::polymorphic_downcast<const ConvPostOpInfoActivation *>(post_op_info.get());
                post_ops.template push_back_op<experimental::PostOpAct<typename TargetInfo::TensorType *>>(act_info->_act);
                break;
            }
            case PostOpType::Eltwise_Add:
            {
                typename TargetInfo::TensorType *add_input    = get_backing_tensor<TargetInfo>(node.input(3));
                const auto                       eltwise_info = utils::cast::polymorphic_downcast<const ConvPostOpInfoEltwiseAdd *>(post_op_info.get());
                post_ops.template push_back_op<experimental::PostOpEltwiseAdd<typename TargetInfo::TensorType *>>(add_input, eltwise_info->_prev_op_dst_pos, eltwise_info->_policy);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported PostOpType");
            }
        }
    }

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;

    using FType = FusedConvolutionBatchNormalizationWithPostOpsFunction<TargetInfo, FusedLayerTypes>;

    // Create and configure function
    std::tie(func, func_name) = create_named_memory_managed_function<FType>(
                                    std::string("FusedConvolutionBatchNormalizationLayerWithPostOpsLayer"), mm, input, weights, biases, output, mean, var, beta, gamma, epsilon, conv_info, num_groups, fast_math, post_ops);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Post Ops:" << post_ops
                               << std::endl);
    return std::move(func);
}

/** Create a backend deconvolution layer function
 *
 * @tparam DeconvolutionLayerFunction Backend deconvolution function
 * @tparam TargetInfo                 Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend deconvolution layer function
 */
template <typename DeconvolutionLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_deconvolution_layer(DeconvolutionLayerNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));

    const PadStrideInfo deconv_info = node.deconvolution_info();

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;

    std::tie(func, std::ignore) = create_named_memory_managed_function<DeconvolutionLayerFunction>(
                                      std::string(), mm,
                                      input, weights, biases, output, deconv_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);
    return func;
}

/** Create a backend layer depth-wise convolution function
 *
 * @tparam DepthwiseConvolutionLayerFunctions Backend depthwise convolution function
 * @tparam TargetInfo                         Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend depth-wise convolution layer function
 */
template <typename DepthwiseConvolutionLayer, typename TargetInfo>
std::unique_ptr<IFunction> create_depthwise_convolution_layer(DepthwiseConvolutionLayerNode &node)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));

    const bool is_quantized = is_data_type_quantized_asymmetric(input->info()->data_type());

    if(is_quantized)
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo       conv_info        = node.convolution_info();
    const unsigned int        depth_multiplier = node.depth_multiplier();
    const ActivationLayerInfo fused_act        = node.fused_activation();

    // Create and configure function (we assume that functions have been validated before creation)
    std::unique_ptr<IFunction> func;
    std::string                func_name;

    std::tie(func, func_name) = create_named_function<DepthwiseConvolutionLayer>(
                                    std::string("DepthwiseConvolutionLayer"),
                                    input, weights, biases, output, conv_info, depth_multiplier, fused_act);

    // Log info
    std::ostringstream qss;
    if(is_quantized)
    {
        qss << " Input QuantInfo: " << input->info()->quantization_info()
            << " Weights QuantInfo: " << weights->info()->quantization_info()
            << " Output QuantInfo: " << output->info()->quantization_info();
    }
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << func_name
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Depth multiplier: " << depth_multiplier
                               << qss.str()
                               << (fused_act.enabled() ? " " + to_string(fused_act.activation()) : "")
                               << std::endl);
    return std::move(func);
}

/** Create a backend depth to space layer function
 *
 * @tparam DepthToSpaceLayerNode Function Backend depth to space function
 * @tparam TargetInfo            Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend depth to space layer function
 */
template <typename DepthToSpaceLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_depth_to_space_layer(DepthToSpaceLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<DepthToSpaceLayerFunction>();
    func->configure(input, output, node.block_shape());

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Block Size: " << node.block_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend dequantize layer function
 *
 * @tparam DequantizationLayer Function Backend dequantize function
 * @tparam TargetInfo          Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend dequantize layer function
 */
template <typename DequantizationLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_dequantization_layer(DequantizationLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<DequantizationLayerFunction>();
    func->configure(input, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Input quantization info: " << output->info()->quantization_info()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}
/** Create a backend detection output layer function
 *
 * @tparam DetectionOutputLayer Function Backend detection output function
 * @tparam TargetInfo           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend detection output layer function
 */
template <typename DetectionOutputLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_detection_output_layer(DetectionOutputLayerNode &node)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input0      = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *input1      = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *input2      = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output      = get_backing_tensor<TargetInfo>(node.output(0));
    const DetectionOutputLayerInfo   detect_info = node.detection_output_info();

    ARM_COMPUTE_ERROR_ON(input0 == nullptr);
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(input2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<DetectionOutputLayerFunction>();
    func->configure(input0, input1, input2, output, detect_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input0->info()->data_type()
                               << " Input0 shape: " << input0->info()->tensor_shape()
                               << " Input1 shape: " << input1->info()->tensor_shape()
                               << " Input2 shape: " << input2->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " DetectionOutputLayer info: " << detect_info
                               << std::endl);

    return std::move(func);
}

/** Create a backend detection post process layer function
 *
 * @tparam DetectionPostProcessLayerFunction Backend detection output function
 * @tparam TargetInfo                        Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend detection post process layer function
 */
template <typename DetectionPostProcessLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_detection_post_process_layer(DetectionPostProcessLayerNode &node)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 4 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType    *input0      = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType    *input1      = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType    *input2      = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType    *output0     = get_backing_tensor<TargetInfo>(node.output(0));
    typename TargetInfo::TensorType    *output1     = get_backing_tensor<TargetInfo>(node.output(1));
    typename TargetInfo::TensorType    *output2     = get_backing_tensor<TargetInfo>(node.output(2));
    typename TargetInfo::TensorType    *output3     = get_backing_tensor<TargetInfo>(node.output(3));
    const DetectionPostProcessLayerInfo detect_info = node.detection_post_process_info();

    ARM_COMPUTE_ERROR_ON(input0 == nullptr);
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(input2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output0 == nullptr);
    ARM_COMPUTE_ERROR_ON(output1 == nullptr);
    ARM_COMPUTE_ERROR_ON(output2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output3 == nullptr);

    // Create and configure function
    auto func = std::make_unique<DetectionPostProcessLayerFunction>();
    func->configure(input0, input1, input2, output0, output1, output2, output3, detect_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input0->info()->data_type()
                               << " Input0 shape: " << input0->info()->tensor_shape()
                               << " Input1 shape: " << input1->info()->tensor_shape()
                               << " Input2 shape: " << input2->info()->tensor_shape()
                               << " Output0 shape: " << output0->info()->tensor_shape()
                               << " Output1 shape: " << output1->info()->tensor_shape()
                               << " Output2 shape: " << output2->info()->tensor_shape()
                               << " Output3 shape: " << output3->info()->tensor_shape()
                               << " DetectionPostProcessLayer info: " << detect_info
                               << std::endl);

    return std::move(func);
}

/** Create a backend element-wise operation layer function
 *
 * @tparam EltwiseFunctions Backend element-wise function
 * @tparam TargetInfo       Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend element-wise operation layer function
 */
template <typename EltwiseFunctions, typename TargetInfo>
std::unique_ptr<IFunction> create_eltwise_layer(EltwiseLayerNode &node)
{
    validate_node<TargetInfo>(node, 2 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input1         = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *input2         = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *output         = get_backing_tensor<TargetInfo>(node.output(0));
    const EltwiseOperation           eltwise_op     = node.eltwise_operation();
    const ConvertPolicy              convert_policy = node.convert_policy();
    const ActivationLayerInfo        act_info       = node.fused_activation();
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(input2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    std::unique_ptr<IFunction> func = nullptr;
    std::string                func_name;
    if(eltwise_op == EltwiseOperation::Add)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Addition>(
                                        std::string("ArithmeticAddition"),
                                        input1, input2, output, convert_policy, act_info);
    }
    else if(eltwise_op == EltwiseOperation::Sub)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Subtraction>(
                                        std::string("ArithmeticSubtraction"),
                                        input1, input2, output, convert_policy, act_info);
    }
    else if(eltwise_op == EltwiseOperation::Mul)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Multiplication>(
                                        std::string("PixelWiseMultiplication"),
                                        input1, input2, output, 1.f, convert_policy, node.rounding_policy(), act_info);
    }
    else if(eltwise_op == EltwiseOperation::Max)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Maximum>(
                                        std::string("ElementwiseMaximum"),
                                        input1, input2, output, act_info);
    }
    else if(eltwise_op == EltwiseOperation::Div)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Division>(
                                        std::string("ArithmeticDivision"),
                                        input1, input2, output, act_info);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported element-wise operation!");
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Operation: " << func_name
                               << " Data Type: " << input1->info()->data_type()
                               << " Shape: " << input1->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend unary element-wise operation layer function
 *
 * @tparam UnaryEltwiseFunctions Backend unary element-wise function
 * @tparam TargetInfo       Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend unary element-wise operation layer function
 */
template <typename UnaryEltwiseFunctions, typename TargetInfo>
std::unique_ptr<IFunction> create_unary_eltwise_layer(UnaryEltwiseLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input      = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output     = get_backing_tensor<TargetInfo>(node.output(0));
    const UnaryEltwiseOperation      eltwise_op = node.eltwise_descriptor().op;

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    std::unique_ptr<IFunction> func = nullptr;
    std::string                func_name;
    if(eltwise_op == UnaryEltwiseOperation::Exp)
    {
        std::tie(func, func_name) = create_named_function<typename UnaryEltwiseFunctions::Exp>(
                                        std::string("Exp"),
                                        input, output);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported unary element-wise operation!");
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Operation: " << func_name
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend flatten layer function
 *
 * @tparam FlattenLayerFunction Backend flatten function
 * @tparam TargetInfo           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend flatten layer function
 */
template <typename FlattenLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_flatten_layer(FlattenLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<FlattenLayerFunction>();
    func->configure(input, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend fully connected layer function
 *
 * @tparam FullyConnectedLayerFunction Backend fully-connected function
 * @tparam TargetInfo                  Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend fully connected layer function
 */
template <typename FullyConnectedLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_fully_connected_layer(FullyConnectedLayerNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));
    FullyConnectedLayerInfo          fc_info = node.info();
    fc_info.enable_fast_math                 = (node.fast_math_hint() == FastMathHint::Enabled);

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(weights == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto wm   = get_weights_manager(ctx, TargetInfo::TargetType);
    auto mm   = get_memory_manager(ctx, TargetInfo::TargetType);
    auto func = std::make_unique<FullyConnectedLayerFunction>(mm, wm.get());
    func->configure(input, weights, biases, output, fc_info);

    const bool is_quantized = is_data_type_quantized_asymmetric(input->info()->data_type());

    // Log info
    std::ostringstream qss;
    if(is_quantized)
    {
        qss << " Input QuantInfo: " << input->info()->quantization_info()
            << " Weights QuantInfo: " << weights->info()->quantization_info()
            << " Output QuantInfo: " << output->info()->quantization_info();
    }
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << qss.str()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend generate proposals layer function
 *
 * @tparam GenerateProposalsLayerFunction Backend generate proposals function
 * @tparam TargetInfo                     Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend generate proposals layer function
 */
template <typename GenerateProposalsLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_generate_proposals_layer(GenerateProposalsLayerNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 3 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *scores              = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *deltas              = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *anchors             = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *proposals           = get_backing_tensor<TargetInfo>(node.output(0));
    typename TargetInfo::TensorType *scores_out          = get_backing_tensor<TargetInfo>(node.output(1));
    typename TargetInfo::TensorType *num_valid_proposals = get_backing_tensor<TargetInfo>(node.output(2));
    const GenerateProposalsInfo      info                = node.info();

    ARM_COMPUTE_ERROR_ON(scores == nullptr);
    ARM_COMPUTE_ERROR_ON(deltas == nullptr);
    ARM_COMPUTE_ERROR_ON(anchors == nullptr);
    ARM_COMPUTE_ERROR_ON(proposals == nullptr);
    ARM_COMPUTE_ERROR_ON(scores_out == nullptr);

    // Create and configure function
    auto func = std::make_unique<GenerateProposalsLayerFunction>(get_memory_manager(ctx, TargetInfo::TargetType));
    func->configure(scores, deltas, anchors, proposals, scores_out, num_valid_proposals, info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << scores->info()->data_type()
                               << " Scores shape: " << scores->info()->tensor_shape()
                               << " Deltas shape: " << deltas->info()->tensor_shape()
                               << " Anchors shape: " << anchors->info()->tensor_shape()
                               << " Proposals shape: " << proposals->info()->tensor_shape()
                               << " Num valid proposals shape: " << num_valid_proposals->info()->tensor_shape()
                               << " Scores Out shape: " << scores_out->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend l2 normalization layer function
 *
 * @tparam NormalizationLayerFunction Backend normalization function
 * @tparam TargetInfo                 Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend normalization layer function
 */
template <typename L2NormalizeLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_l2_normalize_layer(L2NormalizeLayerNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));
    int                              axis    = node.axis();
    float                            epsilon = node.epsilon();

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto mm   = get_memory_manager(ctx, TargetInfo::TargetType);
    auto func = std::make_unique<L2NormalizeLayerFunction>(mm);
    func->configure(input, output, axis, epsilon);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Axis: " << axis
                               << " Epsilon: " << epsilon
                               << std::endl);

    return std::move(func);
}

/** Create a backend normalization layer function
 *
 * @tparam NormalizationLayerFunction Backend normalization function
 * @tparam TargetInfo                 Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend normalization layer function
 */
template <typename NormalizationLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_normalization_layer(NormalizationLayerNode &node, GraphContext &ctx)
{
    ARM_COMPUTE_UNUSED(ctx);

    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input     = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output    = get_backing_tensor<TargetInfo>(node.output(0));
    const NormalizationLayerInfo     norm_info = node.normalization_info();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<NormalizationLayerFunction>();
    func->configure(input, output, norm_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Normalization info: " << norm_info.type()
                               << std::endl);

    return std::move(func);
}

/** Create a backend normalize planar YUV layer function
 *
 * @tparam NormalizePlanarYUVLayerFunction Backend normalize planar YUV function
 * @tparam TargetInfo                      Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend normalize plnar YUV layer function
 */
template <typename NormalizePlanarYUVLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_normalize_planar_yuv_layer(NormalizePlanarYUVLayerNode &node)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *mean   = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *std    = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(mean == nullptr);
    ARM_COMPUTE_ERROR_ON(std == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<NormalizePlanarYUVLayerFunction>();
    func->configure(input, output, mean, std);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend pad layer function
 *
 * @tparam PadLayerFunction Backend pad function
 * @tparam TargetInfo       Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend pad layer function
 */
template <typename PadLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_pad_layer(PadLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input     = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output    = get_backing_tensor<TargetInfo>(node.output(0));
    const PaddingList               &padding   = node.padding();
    const PixelValue                 pad_value = node.pad_value();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<PadLayerFunction>();
    func->configure(input, output, padding, pad_value);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend permute layer function
 *
 * @tparam PermuteLayerFunction Backend permute function
 * @tparam TargetInfo           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend permute layer function
 */
template <typename PermuteLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_permute_layer(PermuteLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    const PermutationVector         &perm   = node.permutation_vector();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<PermuteLayerFunction>();
    func->configure(input, output, perm);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Permutation vector: " << perm
                               << std::endl);

    return std::move(func);
}

/** Create a backend pooling layer function
 *
 * @tparam PoolingLayerFunction Backend pooling function
 * @tparam TargetInfo           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend pooling layer function
 */
template <typename PoolingLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_pooling_layer(PoolingLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input     = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output    = get_backing_tensor<TargetInfo>(node.output(0));
    const PoolingLayerInfo           pool_info = node.pooling_info();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<PoolingLayerFunction>();
    func->configure(input, output, pool_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Pooling info: " << pool_info.pool_type
                               << std::endl);

    return std::move(func);
}

/** Create a backend PRelu layer function
 *
 * @tparam PReluFunction Backend PRelu function
 * @tparam TargetInfo    Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend PRelu layer function
 */
template <typename PReluFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_prelu_layer(PReluLayerNode &node)
{
    validate_node<TargetInfo>(node, 2 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *alpha  = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr || alpha == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<PReluFunction>();
    func->configure(input, alpha, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend print layer function
 *
 * @tparam TargetInfo Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend print layer function
 */
template <typename TargetInfo>
std::unique_ptr<IFunction> create_print_layer(PrintLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    typename TargetInfo::TensorType *input = get_backing_tensor<TargetInfo>(node.input(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_UNUSED(input);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << std::endl);

    return nullptr;
}

/** Create a backend priorbox layer function
 *
 * @tparam PriorBoxLayerFunction Backend priorbox function
 * @tparam TargetInfo           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend priorbox layer function
 */
template <typename PriorBoxLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_priorbox_layer(PriorBoxLayerNode &node)
{
    validate_node<TargetInfo>(node, 2 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input0     = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *input1     = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *output     = get_backing_tensor<TargetInfo>(node.output(0));
    const PriorBoxLayerInfo          prior_info = node.priorbox_info();
    ARM_COMPUTE_ERROR_ON(input0 == nullptr);
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<PriorBoxLayerFunction>();
    func->configure(input0, input1, output, prior_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input0->info()->data_type()
                               << " Input0 shape: " << input0->info()->tensor_shape()
                               << " Input1 shape: " << input1->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " PriorBoxLayer info: " << prior_info
                               << std::endl);

    return std::move(func);
}

/** Create a backend quantization layer function
 *
 * @tparam QuantizationLayerFunction Backend quantization function
 * @tparam TargetInfo                Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend quantization layer function
 */
template <typename QuantizationLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_quantization_layer(QuantizationLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<QuantizationLayerFunction>();
    func->configure(input, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend reduction operation layer function
 *
 * @tparam ReductionOperationFunction Backend reduction operation function
 * @tparam TargetInfo                 Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend reduction sum layer function
 */
template <typename ReductionOperationFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_reduction_operation_layer(ReductionLayerNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input     = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output    = get_backing_tensor<TargetInfo>(node.output(0));
    ReductionOperation               op        = node.op();
    int                              axis      = node.axis();
    bool                             keep_dims = node.keep_dims();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<ReductionOperationFunction>(get_memory_manager(ctx, TargetInfo::TargetType));
    func->configure(input, output, axis, op, keep_dims);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Operation: " << op
                               << " Axis: " << axis
                               << " Keep dimensions:" << keep_dims
                               << std::endl);

    return std::move(func);
}

/** Create a backend reorg layer function
 *
 * @tparam ReorgLayerFunction Backend reorg function
 * @tparam TargetInfo         Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend reshape layer function
 */
template <typename ReorgLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_reorg_layer(ReorgLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<ReorgLayerFunction>();
    func->configure(input, output, node.stride());

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend reshape layer function
 *
 * @tparam ReshapeLayerFunction Backend reshape function
 * @tparam TargetInfo           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend reshape layer function
 */
template <typename ReshapeLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_reshape_layer(ReshapeLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<ReshapeLayerFunction>();
    func->configure(input, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend resize layer function
 *
 * @tparam ResizeLayerFunction Backend resize function
 * @tparam TargetInfo          Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend resize layer function
 */
template <typename ResizeLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_resize_layer(ResizeLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    const InterpolationPolicy policy = node.policy();

    // Create and configure function
    auto func = std::make_unique<ResizeLayerFunction>();
    func->configure(input, output, ScaleKernelInfo{ policy, BorderMode::CONSTANT, PixelValue(), SamplingPolicy::CENTER, false, false });

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Interpolation: " << policy
                               << std::endl);

    return std::move(func);
}

/** Create a backend ROI align layer function
 *
 * @tparam ROIAlignLayerFunction    ROI Align function
 * @tparam TargetInfo               Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return ROI Align layer function
 */
template <typename ROIAlignLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_roi_align_layer(ROIAlignLayerNode &node)
{
    validate_node<TargetInfo>(node, 2 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *rois   = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    ARM_COMPUTE_ERROR_ON(rois == nullptr);

    const ROIPoolingLayerInfo pool_info = node.pooling_info();

    // Create and configure function
    auto func = std::make_unique<ROIAlignLayerFunction>();

    func->configure(input, rois, output, pool_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " ROIs shape: " << rois->info()->tensor_shape()
                               << " ROIPooling width: " << pool_info.pooled_width()
                               << " ROIPooling height: " << pool_info.pooled_height()
                               << std::endl);

    return std::move(func);
}

/** Create a backend slice layer function
 *
 * @tparam SliceLayerFunction Backend slice function
 * @tparam TargetInfo         Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend slice layer function
 */
template <typename SliceLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_slice_layer(SliceLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<SliceLayerFunction>();
    func->configure(input, output, node.starts(), node.ends());

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend softmax layer function
 *
 * @tparam SoftmaxLayerFunction Backend softmax function
 * @tparam TargetInfo           Target-specific information
 *
 * @param[in] node Node to create the backend function for
 * @param[in] ctx  Graph context
 *
 * @return Backend softmax layer function
 */
template <typename SoftmaxLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_softmax_layer(SoftmaxLayerNode &node, GraphContext &ctx)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input  = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    const float                      beta   = node.beta();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<SoftmaxLayerFunction>(get_memory_manager(ctx, TargetInfo::TargetType));
    func->configure(input, output, beta);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend layer stack function
 *
 * @tparam StackLayerFunction Backend stack function
 * @tparam TargetInfo         Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend stack layer function
 */
template <typename StackLayerFunction, typename TargetInfo>
std::unique_ptr<arm_compute::IFunction> create_stack_layer(StackLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating Stack node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    std::vector<typename TargetInfo::TensorType *> inputs;
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        inputs.push_back(get_backing_tensor<TargetInfo>(node.input(i)));
    }
    typename TargetInfo::TensorType *output = get_backing_tensor<TargetInfo>(node.output(0));
    const int                        axis   = node.axis();

    // Create and configure function
    auto func = std::make_unique<StackLayerFunction>();
    func->configure(inputs, axis, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << output->info()->data_type()
                               << " Inputs shape: " << inputs[0]->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Num Inputs: " << inputs.size()
                               << " Axis: " << axis
                               << std::endl);

    return std::move(func);
}

/** Create a backend slice layer function
 *
 * @tparam StridedSliceLayerFunction Backend strided slice function
 * @tparam TargetInfo                Target-specific information
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend strided slice layer function
 */
template <typename StridedSliceLayerFunction, typename TargetInfo>
std::unique_ptr<IFunction> create_strided_slice_layer(StridedSliceLayerNode &node)
{
    validate_node<TargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));
    Coordinates                      starts  = node.starts();
    Coordinates                      ends    = node.ends();
    BiStrides                        strides = node.strides();
    StridedSliceLayerInfo            info    = node.strided_slice_info();

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = std::make_unique<StridedSliceLayerFunction>();
    func->configure(input, output, starts, ends, strides, info.begin_mask(), info.end_mask(), info.shrink_axis_mask());

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated "
                               << node.name()
                               << " Type: " << node.type()
                               << " Target: " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}
} // namespace detail
} // namespace backends
} // namespace graph
} // namespace arm_compute

#endif /* ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_FUNCTION_HELPERS_H */
