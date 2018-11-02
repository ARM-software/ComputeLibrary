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
#include "arm_compute/graph/backends/NEON/NEFunctionFactory.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/backends/Utils.h"
#include "arm_compute/graph/nodes/Nodes.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
namespace
{
/** Returns backing tensor of a given tensor
 *
 * @param[in] tensor Tensor to extract the backing tensor from
 *
 * @return Backing tensor if present else nullptr
 */
arm_compute::ITensor *get_backing_tensor(arm_compute::graph::Tensor *tensor)
{
    return ((tensor == nullptr) || (tensor->handle() == nullptr)) ? nullptr : &tensor->handle()->tensor();
}

/** Create a backend activation layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend activation layer function
 */
std::unique_ptr<IFunction> create_activation_layer(ActivationLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON ActivationLayerNode node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor                  *input    = get_backing_tensor(node.input(0));
    ITensor                  *output   = get_backing_tensor(node.output(0));
    const ActivationLayerInfo act_info = node.activation_info();

    // Create function
    auto func = support::cpp14::make_unique<NEActivationLayer>();
    func->configure(input, output, act_info);

    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NEActivationLayer"
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
 * @param[in] node Node to create the backend function for
 *
 * @return Backend batch normalization layer function
 */
std::unique_ptr<IFunction> create_batch_normalization_layer(BatchNormalizationLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON BatchNormalization node with ID : " << node.id() << " and Name: " << node.name() << std::endl);

    // TODO (geopin01) : Var and mean are compulsory, switch function to accept nullptr as beta and/or gamma
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 5);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor                  *input     = get_backing_tensor(node.input(0));
    ITensor                  *mean      = get_backing_tensor(node.input(1));
    ITensor                  *var       = get_backing_tensor(node.input(2));
    ITensor                  *beta      = get_backing_tensor(node.input(3));
    ITensor                  *gamma     = get_backing_tensor(node.input(4));
    ITensor                  *output    = get_backing_tensor(node.output(0));
    const float               epsilon   = node.epsilon();
    const ActivationLayerInfo fused_act = node.fused_activation();

    // Create and configure function
    auto func = support::cpp14::make_unique<NEBatchNormalizationLayer>();
    func->configure(input, output, mean, var, beta, gamma, epsilon, fused_act);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NEBatchNormalizationLayer"
                               << " Data Type: " << input->info()->data_type()
                               << " Shape: " << input->info()->tensor_shape()
                               << " Epsilon: " << epsilon << " "
                               << (fused_act.enabled() ? to_string(fused_act.activation()) : "")
                               << " InPlace : " << is_in_place_operation(input, output)
                               << std::endl);

    return std::move(func);
}

/** Create a backend convolution layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend convolution layer function
 */
std::unique_ptr<IFunction> create_convolution_layer(ConvolutionLayerNode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON ConvolutionLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor *input   = get_backing_tensor(node.input(0));
    ITensor *weights = get_backing_tensor(node.input(1));
    ITensor *biases  = get_backing_tensor(node.input(2));
    ITensor *output  = get_backing_tensor(node.output(0));

    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo     conv_info      = node.convolution_info();
    const ConvolutionMethod conv_algorithm = node.convolution_method();

    // Create and configure function (we assume that functions have been validated before creation)
    std::shared_ptr<IMemoryManager> mm = get_memory_manager(ctx, Target::NEON);
    std::unique_ptr<IFunction>      func;
    std::string                     func_name;
    if(conv_algorithm == ConvolutionMethod::DIRECT)
    {
        std::tie(func, func_name) = create_named_memory_managed_function<NEDirectConvolutionLayer>(std::string("NEDirectConvolutionLayer"), mm,
                                                                                                   input, weights, biases, output, conv_info);
    }
    else if(conv_algorithm == ConvolutionMethod::GEMM)
    {
        std::tie(func, func_name) = create_named_memory_managed_function<NEGEMMConvolutionLayer>(std::string("NEGEMMConvolutionLayer"), mm,
                                                                                                 input, weights, biases, output, conv_info);
    }
    else if(conv_algorithm == ConvolutionMethod::WINOGRAD)
    {
        std::tie(func, func_name) = create_named_memory_managed_function<NEWinogradConvolutionLayer>(std::string("NEWinogradConvolutionLayer"), mm,
                                                                                                     input, weights, biases, output, conv_info);
    }
    else
    {
        std::tie(func, func_name) = create_named_memory_managed_function<NEConvolutionLayer>(std::string("NEConvolutionLayer"), mm,
                                                                                             input, weights, biases, output, conv_info);
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << func_name
                               << " Data Type: " << input->info()->data_type()
                               << " Input QuantInfo: " << input->info()->quantization_info()
                               << " Weights QuantInfo: " << weights->info()->quantization_info()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);
    return func;
}

/** Create a backend layer depth concatenate function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend depth concatenate layer function
 */
std::unique_ptr<arm_compute::IFunction> create_depth_concatenate_layer(DepthConcatenateLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON DepthConcatenate node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Return nullptr if depth concatenate is switched off
    if(!node.is_enabled())
    {
        return nullptr;
    }

    // Extract IO and info
    std::vector<arm_compute::ITensor *> inputs;
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        inputs.push_back(get_backing_tensor(node.input(i)));
    }
    ITensor *output = get_backing_tensor(node.output(0));

    // Create and configure function
    auto func = support::cpp14::make_unique<NEDepthConcatenateLayer>();
    func->configure(inputs, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NEDepthConcatenateLayer"
                               << " Data Type: " << output->info()->data_type()
                               << " Shape: " << output->info()->tensor_shape()
                               << " Num Inputs: " << inputs.size()
                               << std::endl);

    return std::move(func);
}

/** Create a backend layer depth-wise convolution function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend depth-wise convolution layer function
 */
std::unique_ptr<IFunction> create_depthwise_convolution_layer(DepthwiseConvolutionLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON DepthwiseConvolutionLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor *input   = get_backing_tensor(node.input(0));
    ITensor *weights = get_backing_tensor(node.input(1));
    ITensor *biases  = get_backing_tensor(node.input(2));
    ITensor *output  = get_backing_tensor(node.output(0));

    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        biases->info()->set_data_type(DataType::S32);
    }

    const PadStrideInfo              conv_info     = node.convolution_info();
    const DepthwiseConvolutionMethod dwc_algorithm = node.depthwise_convolution_method();

    // Create and configure function (we assume that functions have been validated before creation)
    std::unique_ptr<IFunction> func;
    std::string                func_name;
    if(dwc_algorithm == DepthwiseConvolutionMethod::OPTIMIZED_3x3)
    {
        std::tie(func, func_name) = create_named_function<NEDepthwiseConvolutionLayer3x3>(std::string("NEDepthwiseConvolutionLayer3x3"),
                                                                                          input, weights, biases, output, conv_info);
    }
    else
    {
        std::tie(func, func_name) = create_named_function<NEDepthwiseConvolutionLayer>(std::string("NEDepthwiseConvolutionLayer"),
                                                                                       input, weights, biases, output, conv_info);
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << func_name
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
 * @param[in] node Node to create the backend function for
 *
 * @return Backend element-wise operation layer function
 */
std::unique_ptr<IFunction> create_eltwise_layer(EltwiseLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON EltwiseLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 2);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor               *input1     = get_backing_tensor(node.input(0));
    ITensor               *input2     = get_backing_tensor(node.input(1));
    ITensor               *output     = get_backing_tensor(node.output(0));
    const EltwiseOperation eltwise_op = node.eltwise_operation();
    ARM_COMPUTE_ERROR_ON(input1 == nullptr);
    ARM_COMPUTE_ERROR_ON(input2 == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    std::unique_ptr<IFunction> func = nullptr;
    std::string                func_name;
    if(eltwise_op == EltwiseOperation::ADD)
    {
        std::tie(func, func_name) = create_named_function<NEArithmeticAddition>(std::string("NEArithmeticAddition"),
                                                                                input1, input2, output, ConvertPolicy::SATURATE);
    }
    else if(eltwise_op == EltwiseOperation::SUB)
    {
        std::tie(func, func_name) = create_named_function<NEArithmeticSubtraction>(std::string("NEArithmeticSubtraction"),
                                                                                   input1, input2, output, ConvertPolicy::SATURATE);
    }
    else if(eltwise_op == EltwiseOperation::MUL)
    {
        std::tie(func, func_name) = create_named_function<NEPixelWiseMultiplication>(std::string("NEPixelWiseMultiplication"),
                                                                                     input1, input2, output, 1.f,
                                                                                     ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported element-wise operation!");
    }

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated " << func_name
                               << " Data Type: " << input1->info()->data_type()
                               << " Shape : " << input1->info()->tensor_shape()
                               << std::endl);

    return func;
}

/** Create a backend flatten layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend flatten layer function
 */
std::unique_ptr<IFunction> create_flatten_layer(FlattenLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON FlattenLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor *input  = get_backing_tensor(node.input(0));
    ITensor *output = get_backing_tensor(node.output(0));

    // Create and configure function
    auto func = support::cpp14::make_unique<NEFlattenLayer>();
    func->configure(input, output);
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NEFlattenLayer"
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend fully connected layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend fully connected layer function
 */
std::unique_ptr<IFunction> create_fully_connected_layer(FullyConnectedLayerNode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON FullyConnectedLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 3);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor *input   = get_backing_tensor(node.input(0));
    ITensor *weights = get_backing_tensor(node.input(1));
    ITensor *biases  = get_backing_tensor(node.input(2));
    ITensor *output  = get_backing_tensor(node.output(0));

    // Create and configure function
    auto func = support::cpp14::make_unique<NEFullyConnectedLayer>(get_memory_manager(ctx, Target::NEON));
    func->configure(input, weights, biases, output);
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(weights == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NEFullyConnectedLayer"
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Weights shape: " << weights->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend normalization layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend normalization layer function
 */
std::unique_ptr<IFunction> create_normalization_layer(NormalizationLayerNode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON NormalizationLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor                     *input     = get_backing_tensor(node.input(0));
    ITensor                     *output    = get_backing_tensor(node.output(0));
    const NormalizationLayerInfo norm_info = node.normalization_info();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = support::cpp14::make_unique<NENormalizationLayer>(get_memory_manager(ctx, Target::NEON));
    func->configure(input, output, norm_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NENormalizationLayer"
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Normalization info: " << norm_info.type()
                               << std::endl);

    return std::move(func);
}

/** Create a backend pooling layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend pooling layer function
 */
std::unique_ptr<IFunction> create_pooling_layer(PoolingLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON PoolingLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor               *input     = get_backing_tensor(node.input(0));
    ITensor               *output    = get_backing_tensor(node.output(0));
    const PoolingLayerInfo pool_info = node.pooling_info();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = support::cpp14::make_unique<NEPoolingLayer>();
    func->configure(input, output, pool_info);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NEPoolingLayer"
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << " Pooling info: " << pool_info.pool_type()
                               << std::endl);

    return std::move(func);
}

/** Create a backend reshape layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend reshape layer function
 */
std::unique_ptr<IFunction> create_reshape_layer(ReshapeLayerNode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON ReshapeLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor *input  = get_backing_tensor(node.input(0));
    ITensor *output = get_backing_tensor(node.output(0));
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = support::cpp14::make_unique<NEReshapeLayer>();
    func->configure(input, output);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NEReshapeLayer"
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}

/** Create a backend softmax layer function
 *
 * @param[in] node Node to create the backend function for
 *
 * @return Backend softmax layer function
 */
std::unique_ptr<IFunction> create_softmax_layer(SoftmaxLayerNode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Creating NEON SoftmaxLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.num_inputs() != 1);
    ARM_COMPUTE_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    ITensor    *input  = get_backing_tensor(node.input(0));
    ITensor    *output = get_backing_tensor(node.output(0));
    const float beta   = node.beta();
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    // Create and configure function
    auto func = support::cpp14::make_unique<NESoftmaxLayer>(get_memory_manager(ctx, Target::NEON));
    func->configure(input, output, beta);

    // Log info
    ARM_COMPUTE_LOG_GRAPH_INFO("Instantiated NESoftmaxLayer"
                               << " Data Type: " << input->info()->data_type()
                               << " Input shape: " << input->info()->tensor_shape()
                               << " Output shape: " << output->info()->tensor_shape()
                               << std::endl);

    return std::move(func);
}
} // namespace

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
            return create_activation_layer(*polymorphic_downcast<ActivationLayerNode *>(node));
        case NodeType::BatchNormalizationLayer:
            return create_batch_normalization_layer(*polymorphic_downcast<BatchNormalizationLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return create_convolution_layer(*polymorphic_downcast<ConvolutionLayerNode *>(node), ctx);
        case NodeType::DepthConcatenateLayer:
            return create_depth_concatenate_layer(*polymorphic_downcast<DepthConcatenateLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return create_depthwise_convolution_layer(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::EltwiseLayer:
            return create_eltwise_layer(*polymorphic_downcast<EltwiseLayerNode *>(node));
        case NodeType::FlattenLayer:
            return create_flatten_layer(*polymorphic_downcast<FlattenLayerNode *>(node));
        case NodeType::FullyConnectedLayer:
            return create_fully_connected_layer(*polymorphic_downcast<FullyConnectedLayerNode *>(node), ctx);
        case NodeType::NormalizationLayer:
            return create_normalization_layer(*polymorphic_downcast<NormalizationLayerNode *>(node), ctx);
        case NodeType::PoolingLayer:
            return create_pooling_layer(*polymorphic_downcast<PoolingLayerNode *>(node));
        case NodeType::ReshapeLayer:
            return create_reshape_layer(*polymorphic_downcast<ReshapeLayerNode *>(node));
        case NodeType::SoftmaxLayer:
            return create_softmax_layer(*polymorphic_downcast<SoftmaxLayerNode *>(node), ctx);
        default:
            return nullptr;
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute