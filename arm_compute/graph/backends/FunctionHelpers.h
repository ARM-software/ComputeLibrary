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
#ifndef __ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_FUNCTION_HELPERS_H__
#define __ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_FUNCTION_HELPERS_H__

#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/graph/backends/Utils.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/utils/misc/Cast.h"

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
                                  << " Target : " << TargetInfo::TargetType
                                  << " ID : " << node.id()
                                  << " Name: " << node.name()
                                  << std::endl);

    ARM_COMPUTE_ERROR_ON(TargetInfo::TargetType != node.assigned_target());
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != num_expected_inputs);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != num_expected_outputs);
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
    auto func = support::cpp14::make_unique<ActivationLayerFunction>();
    func->configure(input, output, act_info);

    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " Activation function: " << act_info.activation()
                               << " a: " << act_info.a()
                               << " b: " << act_info.b()
                               << " InPlace : " << is_in_place_operation(input, output)
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
    typename TargetInfo::TensorType *input     = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *mean      = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *var       = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *beta      = get_backing_tensor<TargetInfo>(node.input(3));
    typename TargetInfo::TensorType *gamma     = get_backing_tensor<TargetInfo>(node.input(4));
    typename TargetInfo::TensorType *output    = get_backing_tensor<TargetInfo>(node.output(0));
    const float                      epsilon   = node.epsilon();
    const ActivationLayerInfo        fused_act = node.fused_activation();

    // Create and configure function
    auto func = support::cpp14::make_unique<BatchNormalizationLayerFunction>();
    func->configure(input, output, mean, var, beta, gamma, epsilon, fused_act);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " Epsilon: " << epsilon << " "
                               << (fused_act.enabled() ? to_string(fused_act.activation()) : "")
                               << " InPlace : " << is_in_place_operation(input, output)
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
    auto func = support::cpp14::make_unique<ChannelShuffleLayerFunction>();
    func->configure(input, output, num_groups);

    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
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
    std::vector<typename TargetInfo::TensorType *> inputs;
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        inputs.push_back(get_backing_tensor<TargetInfo>(node.input(i)));
    }
    typename TargetInfo::TensorType *output      = get_backing_tensor<TargetInfo>(node.output(0));
    const DataLayoutDimension        concat_axis = node.concatenation_axis();

    // Create and configure function
    auto func = support::cpp14::make_unique<ConcatenateLayerFunction>();
    func->configure(inputs, output, concat_axis);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << output->info()->data_type()
                               << " Shape: " << output->info()->tensor_shape()
                               << " Num Inputs: " << inputs.size()
                               << " Axis: " << concat_axis
                               << std::endl);

    return std::move(func);
}

/** Create a backend convolution layer function
 *
 * @tparam ConvolutionLayerFunctions Backend convolution functions
 * @tparam TargetInfo              Target-specific information
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

    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo     conv_info      = node.convolution_info();
    const ConvolutionMethod conv_algorithm = node.convolution_method();
    const bool              fast_math      = node.fast_math_hint() == FastMathHint::Enabled;

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;

    if(conv_algorithm == ConvolutionMethod::Winograd)
    {
        std::tie(func, func_name) = create_named_memory_managed_function<typename ConvolutionLayerFunctions::WinogradConvolutionLayer>(
                                        std::string("WinogradConvolutionLayer"), mm,
                                        input, weights, biases, output, conv_info, ActivationLayerInfo(), fast_math);
    }
    else if(conv_algorithm == ConvolutionMethod::Direct)
    {
        std::tie(func, func_name) = create_named_function<typename ConvolutionLayerFunctions::DirectConvolutionLayer>(
                                        std::string("DirectConvolutionLayer"),
                                        input, weights, biases, output, conv_info);
    }
    else if(conv_algorithm == ConvolutionMethod::GEMM)
    {
        std::tie(func, func_name) = create_named_memory_managed_function<typename ConvolutionLayerFunctions::GEMMConvolutionLayer>(
                                        std::string("GEMMConvolutionLayer"), mm,
                                        input, weights, biases, output, conv_info);
    }
    else
    {
        std::tie(func, func_name) = create_named_memory_managed_function<typename ConvolutionLayerFunctions::GenericConvolutionLayer>(
                                        std::string("GenericConvolutionLayer"), mm,
                                        input, weights, biases, output, conv_info, WeightsInfo(), Size2D(1U, 1U), ActivationLayerInfo(), fast_math);
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << func_name
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input QuantInfo: " << input->info()->quantization_info()
                               << " Weights QuantInfo: " << weights->info()->quantization_info()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);
    return func;
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

    const PadStrideInfo deconv_info  = node.deconvolution_info();
    const Size2D        inner_border = node.inner_border();

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, TargetInfo::TargetType);
    std::unique_ptr<IFunction>      func;

    std::tie(func, std::ignore) = create_named_memory_managed_function<DeconvolutionLayerFunction>(
                                      std::string(), mm,
                                      input, weights, biases, output, deconv_info, inner_border.x(), inner_border.y());

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
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
template <typename DepthwiseConvolutionLayerFunctions, typename TargetInfo>
std::unique_ptr<IFunction> create_depthwise_convolution_layer(DepthwiseConvolutionLayerNode &node)
{
    validate_node<TargetInfo>(node, 3 /* expected inputs */, 1 /* expected outputs */);

    // Extract IO and info
    typename TargetInfo::TensorType *input   = get_backing_tensor<TargetInfo>(node.input(0));
    typename TargetInfo::TensorType *weights = get_backing_tensor<TargetInfo>(node.input(1));
    typename TargetInfo::TensorType *biases  = get_backing_tensor<TargetInfo>(node.input(2));
    typename TargetInfo::TensorType *output  = get_backing_tensor<TargetInfo>(node.output(0));

    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo              conv_info     = node.convolution_info();
    const DepthwiseConvolutionMethod dwc_algorithm = node.depthwise_convolution_method();

    // Create and configure function (we assume that functions have been validated before creation)
    std::unique_ptr<IFunction> func;
    std::string                func_name;
    if(dwc_algorithm == DepthwiseConvolutionMethod::Optimized3x3)
    {
        std::tie(func, func_name) = create_named_function<typename DepthwiseConvolutionLayerFunctions::DepthwiseConvolutionLayer3x3>(
                                        std::string("DepthwiseConvolutionLayer3x3"),
                                        input, weights, biases, output, conv_info);
    }
    else
    {
        std::tie(func, func_name) = create_named_function<typename DepthwiseConvolutionLayerFunctions::GenericDepthwiseConvolutionLayer>(
                                        std::string("DepthwiseConvolutionLayer"),
                                        input, weights, biases, output, conv_info);
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << func_name
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input QuantInfo: " << input->info()->quantization_info()
                               << " Weights QuantInfo: " << weights->info()->quantization_info()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);
    return func;
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
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(input2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    std::unique_ptr<IFunction> func = nullptr;
    std::string                func_name;
    if(eltwise_op == EltwiseOperation::Add)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Addition>(
                                        std::string("ArithmeticAddition"),
                                        input1, input2, output, convert_policy);
    }
    else if(eltwise_op == EltwiseOperation::Sub)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Subtraction>(
                                        std::string("ArithmeticSubtraction"),
                                        input1, input2, output, convert_policy);
    }
    else if(eltwise_op == EltwiseOperation::Mul)
    {
        std::tie(func, func_name) = create_named_function<typename EltwiseFunctions::Multiplication>(
                                        std::string("PixelWiseMultiplication"),
                                        input1, input2, output, 1.f, convert_policy, node.rounding_policy());
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported element-wise operation!");
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Operation " << func_name
                               << " Data Type: " << input1->info()->data_type()
                               << " Shape : " << input1->info()->tensor_shape()
                               << std::endl);

    return func;
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
    auto func = support::cpp14::make_unique<FlattenLayerFunction>();
    func->configure(input, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
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
    const FullyConnectedLayerInfo    fc_info = node.info();

    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(weights == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = support::cpp14::make_unique<FullyConnectedLayerFunction>(get_memory_manager(ctx, TargetInfo::TargetType));
    func->configure(input, weights, biases, output, fc_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
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
    auto func = support::cpp14::make_unique<NormalizationLayerFunction>();
    func->configure(input, output, norm_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Normalization info: " << norm_info.type()
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
    auto func = support::cpp14::make_unique<PoolingLayerFunction>();
    func->configure(input, output, pool_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Pooling info: " << pool_info.pool_type()
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
    auto func = support::cpp14::make_unique<ReshapeLayerFunction>();
    func->configure(input, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
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
    auto func = support::cpp14::make_unique<ResizeLayerFunction>();
    func->configure(input, output, policy, BorderMode::CONSTANT);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Interpolation: " << policy
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
    auto func = support::cpp14::make_unique<SoftmaxLayerFunction>(get_memory_manager(ctx, TargetInfo::TargetType));
    func->configure(input, output, beta);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << node.type()
                               << " Target " << TargetInfo::TargetType
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

#endif /* __ARM_COMPUTE_GRAPH_BACKENDS_DETAIL_FUNCTION_HELPERS_H__ */
