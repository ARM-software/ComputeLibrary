/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/graph/nodes/ConvolutionLayer.h"

#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphTypePrinter.h"
#include "utils/TypePrinter.h"

#include <tuple>
#include <vector>

using namespace arm_compute::graph;

namespace
{
/** Calculates the output shaped of the convolution layer
 *
 * @param[in] input_shape   Input tensor shape
 * @param[in] weights_shape Weights shape
 * @param[in] conv_info     Convolution information (padding, stride, etc.)
 *
 * @return The expected output tensor shape
 */
TensorShape calculate_convolution_layer_output_shape(const TensorShape &input_shape, const TensorShape &weights_shape, const PadStrideInfo &conv_info)
{
    unsigned int output_width  = 0;
    unsigned int output_height = 0;

    // Get output width and height
    std::tie(output_width, output_height) = arm_compute::scaled_dimensions(input_shape.x(), input_shape.y(), weights_shape.x(), weights_shape.y(), conv_info);

    // Create output shape
    TensorShape output_shape = input_shape;
    output_shape.set(0, output_width);
    output_shape.set(1, output_height);
    output_shape.set(2, weights_shape[3]);

    return output_shape;
}

// Instantiate GEMM based convolution layer
template <typename ConvolutionType, typename TensorType, TargetHint target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(ITensor *input, ITensor *weights, ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    auto conv = arm_compute::support::cpp14::make_unique<ConvolutionType>();
    conv->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(weights),
        dynamic_cast<TensorType *>(biases),
        dynamic_cast<TensorType *>(output),
        conv_info, weights_info);
    return std::move(conv);
}

// Instantiate direct convolution layer
template <typename ConvolutionType, typename TensorType, TargetHint target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate_direct_function(ITensor *input, ITensor *weights, ITensor *biases, ITensor *output, const PadStrideInfo &conv_info)
{
    auto conv = arm_compute::support::cpp14::make_unique<ConvolutionType>();
    conv->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(weights),
        dynamic_cast<TensorType *>(biases),
        dynamic_cast<TensorType *>(output),
        conv_info);
    return std::move(conv);
}

template <TargetHint                    target_hint>
std::unique_ptr<arm_compute::IFunction> instantiate(ITensor *input, ITensor *weights, ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                                    ConvolutionMethodHint conv_method);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::OPENCL>(ITensor *input, ITensor *weights, ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                                                        const WeightsInfo    &weights_info,
                                                                        ConvolutionMethodHint conv_method)
{
    if(conv_method == ConvolutionMethodHint::GEMM)
    {
        return instantiate_function<arm_compute::CLConvolutionLayer, arm_compute::ICLTensor, TargetHint::OPENCL>(input, weights, biases, output, conv_info, weights_info);
    }
    else
    {
        return instantiate_direct_function<arm_compute::CLDirectConvolutionLayer, arm_compute::ICLTensor, TargetHint::OPENCL>(input, weights, biases, output, conv_info);
    }
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<TargetHint::NEON>(ITensor *input, ITensor *weights, ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                                                      const WeightsInfo    &weights_info,
                                                                      ConvolutionMethodHint conv_method)
{
    if(conv_method == ConvolutionMethodHint::GEMM)
    {
        return instantiate_function<arm_compute::NEConvolutionLayer, arm_compute::ITensor, TargetHint::NEON>(input, weights, biases, output, conv_info, weights_info);
    }
    else
    {
        return instantiate_direct_function<arm_compute::NEDirectConvolutionLayer, arm_compute::ITensor, TargetHint::NEON>(input, weights, biases, output, conv_info);
    }
}
} // namespace

/** Grouped Convolution function */
class GroupedConvolutionFunction final : public arm_compute::IFunction
{
public:
    /** Default Constructor */
    GroupedConvolutionFunction()
        : _convolutions()
    {
    }
    /** Default Destructor */
    ~GroupedConvolutionFunction() final = default;
    /** Prevent instances from being copy constructed */
    GroupedConvolutionFunction(const GroupedConvolutionFunction &) = delete;
    /** Prevent instances from being copy assigned */
    GroupedConvolutionFunction &operator=(const GroupedConvolutionFunction &) = delete;
    /** Allow instances to be move constructed */
    GroupedConvolutionFunction(GroupedConvolutionFunction &&) noexcept = default;
    /** Allow instances to be move assigned */
    GroupedConvolutionFunction &operator=(GroupedConvolutionFunction &&) noexcept = default;
    /** Adds a convolution
     *
     * @param convolution Convolution function to add
     */
    void add_convolution_function(std::unique_ptr<IFunction> convolution)
    {
        _convolutions.emplace_back(std::move(convolution));
    }

    // Inherited methods overriden:
    void run() override
    {
        for(auto &c : _convolutions)
        {
            c->run();
        }
    }

private:
    std::vector<std::unique_ptr<IFunction>> _convolutions;
};

std::unique_ptr<arm_compute::IFunction> ConvolutionLayer::instantiate_node(GraphContext &ctx, ITensor *input, ITensor *output)
{
    // Set weights and biases info
    if(_weights.tensor() == nullptr)
    {
        _weights.set_info(TensorInfo(TensorShape(_conv_width, _conv_height, input->info()->dimension(2) / _num_groups, _ofm),
                                     input->info()->num_channels(), input->info()->data_type(),
                                     input->info()->fixed_point_position()));
    }
    if(_biases.tensor() == nullptr)
    {
        _biases.set_info(TensorInfo(TensorShape(_ofm), input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position()));
    }

    std::unique_ptr<arm_compute::IFunction> func;
    _target_hint                                 = ctx.hints().target_hint();
    _input                                       = input;
    _output                                      = output;
    const ConvolutionMethodHint conv_method_hint = ctx.hints().convolution_method_hint();

    // Check if the weights and biases are loaded
    bool weights_are_loaded = _weights.tensor() != nullptr;
    bool biases_are_loaded  = _weights.tensor() != nullptr;

    // Set bias and weights target
    _weights.set_target(_target_hint);
    _biases.set_target(_target_hint);

    // Calculate output shape
    TensorShape output_shape = calculate_convolution_layer_output_shape(_input->info()->tensor_shape(), _weights.info().tensor_shape(), _conv_info);

    // Output auto inizialitation if not yet initialized
    arm_compute::auto_init_if_empty(*_output->info(), output_shape, 1, _input->info()->data_type(), _input->info()->fixed_point_position());

    // Create appropriate convolution function
    if(_num_groups == 1)
    {
        func = instantiate_convolution(conv_method_hint);
    }
    else
    {
        func = instantiate_grouped_convolution(conv_method_hint);
    }

    // Fill weights
    if(!weights_are_loaded)
    {
        _weights.allocate_and_fill_if_needed();
    }
    // Fill biases
    if(!biases_are_loaded)
    {
        _biases.allocate_and_fill_if_needed();
    }

    return func;
}

void ConvolutionLayer::print_info()
{
    if(_target_hint == TargetHint::OPENCL)
    {
        std::cout << "Instantiating CLConvolutionLayer";
    }
    else
    {
        std::cout << "Instantiating NEConvolutionLayer";
    }
    std::cout << " Data Type: " << _input->info()->data_type()
              << " Input Shape: " << _input->info()->tensor_shape()
              << " Weights shape: " << _weights.info().tensor_shape()
              << " Biases Shape: " << _biases.info().tensor_shape()
              << " Output Shape: " << _output->info()->tensor_shape()
              << " PadStrideInfo: " << _conv_info
              << " Groups: " << _num_groups
              << " WeightsInfo: " << _weights_info
              << std::endl;
}

std::unique_ptr<arm_compute::IFunction> ConvolutionLayer::instantiate_convolution(ConvolutionMethodHint conv_method_hint)
{
    std::unique_ptr<arm_compute::IFunction> func;
    if(_target_hint == TargetHint::OPENCL)
    {
        func = instantiate<TargetHint::OPENCL>(_input, _weights.tensor(), _biases.tensor(), _output, _conv_info, _weights_info, conv_method_hint);
    }
    else
    {
        func = instantiate<TargetHint::NEON>(_input, _weights.tensor(), _biases.tensor(), _output, _conv_info, _weights_info, conv_method_hint);
    }
    return func;
}

std::unique_ptr<arm_compute::IFunction> ConvolutionLayer::instantiate_grouped_convolution(ConvolutionMethodHint conv_method_hint)
{
    // Get tensor shapes
    TensorShape input_shape   = _input->info()->tensor_shape();
    TensorShape output_shape  = _output->info()->tensor_shape();
    TensorShape weights_shape = _weights.info().tensor_shape();
    TensorShape biases_shape  = _biases.info().tensor_shape();

    ARM_COMPUTE_ERROR_ON_MSG((input_shape.z() % _num_groups) != 0, "Input depth not multiple of the number of groups!");
    ARM_COMPUTE_ERROR_ON_MSG((output_shape.z() % _num_groups) != 0, "Output depth not multiple of the number of groups!");
    ARM_COMPUTE_ERROR_ON_MSG((weights_shape[3] % _num_groups) != 0, "Number of kernels not multiple of the number of groups!");
    ARM_COMPUTE_ERROR_ON_MSG((biases_shape.x() % _num_groups) != 0, "Biases not multiple of the number of groups!");

    // Create a grouped convolution function
    auto grouped_conv = arm_compute::support::cpp14::make_unique<GroupedConvolutionFunction>();

    // Create sub-tensors vectors
    _is = arm_compute::support::cpp14::make_unique<SubTensor[]>(_num_groups);
    _os = arm_compute::support::cpp14::make_unique<SubTensor[]>(_num_groups);
    _ws = arm_compute::support::cpp14::make_unique<SubTensor[]>(_num_groups);
    _bs = arm_compute::support::cpp14::make_unique<SubTensor[]>(_num_groups);

    // Calculate sub-tensor splits
    const int input_split   = input_shape.z() / _num_groups;
    const int output_split  = output_shape.z() / _num_groups;
    const int weights_split = weights_shape[3] / _num_groups;
    const int biases_split  = biases_shape.x() / _num_groups;

    // Calculate sub-tensor shapes
    input_shape.set(2, input_split);
    output_shape.set(2, output_split);
    weights_shape.set(3, weights_split);
    biases_shape.set(0, biases_split);

    // Configure sub-tensors
    for(int i = 0; i < static_cast<int>(_num_groups); ++i)
    {
        // Create convolution function
        std::unique_ptr<arm_compute::IFunction> func;

        // Calculate sub-tensors starting coordinates
        Coordinates input_coord(0, 0, input_split * i);
        Coordinates output_coord(0, 0, output_split * i);
        Coordinates weights_coord(0, 0, 0, weights_split * i);
        Coordinates biases_coord(biases_split * i);

        // Create sub-tensors for input, output, weights and bias
        auto hint_to_use = (_target_hint == TargetHint::OPENCL) ? TargetHint::OPENCL : TargetHint::NEON;
        _is[i]           = SubTensor(_input, input_shape, input_coord, hint_to_use);
        _os[i]           = SubTensor(_output, output_shape, output_coord, hint_to_use);
        _ws[i]           = SubTensor(_weights.tensor(), weights_shape, weights_coord, hint_to_use);
        _bs[i]           = SubTensor(_biases.tensor(), biases_shape, biases_coord, hint_to_use);

        // Instantiate convolution function
        if(_target_hint == TargetHint::OPENCL)
        {
            func = instantiate<TargetHint::OPENCL>(_is[i].tensor(), _ws[i].tensor(), _bs[i].tensor(), _os[i].tensor(), _conv_info, _weights_info, conv_method_hint);
        }
        else
        {
            func = instantiate<TargetHint::NEON>(_is[i].tensor(), _ws[i].tensor(), _bs[i].tensor(), _os[i].tensor(), _conv_info, _weights_info, conv_method_hint);
        }

        // Add convolution function to the list of convolutions for the grouped convolution
        grouped_conv->add_convolution_function(std::move(func));
    }

    return std::move(grouped_conv);
}
