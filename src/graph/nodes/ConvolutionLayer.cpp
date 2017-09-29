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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename ConvolutionType, typename TensorType, Hint hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    bool weights_are_loaded = weights.tensor() != nullptr;
    bool biases_are_loaded  = biases.tensor() != nullptr;

    auto conv = arm_compute::support::cpp14::make_unique<ConvolutionType>();
    conv->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(weights.set_target(hint)),
        dynamic_cast<TensorType *>(biases.set_target(hint)),
        dynamic_cast<TensorType *>(output),
        conv_info, weights_info);
    if(!weights_are_loaded)
    {
        weights.allocate_and_fill_if_needed();
    }
    if(!biases_are_loaded)
    {
        biases.allocate_and_fill_if_needed();
    }

    return std::move(conv);
}

template <Hint                          hint>
std::unique_ptr<arm_compute::IFunction> instantiate(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<Hint::OPENCL>(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    return instantiate_function<arm_compute::CLConvolutionLayer, arm_compute::CLTensor, Hint::OPENCL>(input, weights, biases, output, conv_info, weights_info);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<Hint::NEON>(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info)
{
    return instantiate_function<arm_compute::NEConvolutionLayer, arm_compute::Tensor, Hint::NEON>(input, weights, biases, output, conv_info, weights_info);
}
} // namespace

std::unique_ptr<arm_compute::IFunction> ConvolutionLayer::instantiate_node(Hint hint, ITensor *input, ITensor *output)
{
    if(_weights.tensor() == nullptr)
    {
        _weights.set_info(TensorInfo(TensorShape(_conv_width, _conv_height, input->info()->dimension(2), _ofm), input->info()->num_channels(), input->info()->data_type(),
                                     input->info()->fixed_point_position()));
    }
    if(_biases.tensor() == nullptr)
    {
        _biases.set_info(TensorInfo(TensorShape(_ofm), input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position()));
    }

    std::unique_ptr<arm_compute::IFunction> func;
    _hint   = hint;
    _input  = input;
    _output = output;

    if(_hint == Hint::OPENCL)
    {
        func = instantiate<Hint::OPENCL>(input, _weights, _biases, output, _conv_info, _weights_info);
    }
    else
    {
        func = instantiate<Hint::NEON>(input, _weights, _biases, output, _conv_info, _weights_info);
    }

    return func;
}

void ConvolutionLayer::print_info()
{
    if(_hint == Hint::OPENCL)
    {
        std::cout << "Instantiating CLConvolutionLayer";
    }
    else
    {
        std::cout << "Instantiating NEConvolutionLayer";
    }
    std::cout << " Type: " << _input->info()->data_type() << " Input Shape: " << _input->info()->tensor_shape() << " Weights shape: " << _weights.info().tensor_shape() << " Biases Shape: " <<
              _biases.info().tensor_shape() << " Output Shape: " << _output->info()->tensor_shape() << " PadStrideInfo: " << _conv_info << "WeightsInfo: " << _weights_info << std::endl;
}
