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
#include "arm_compute/graph/nodes/ActivationLayer.h"

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename ActivationType, typename TensorType, Hint hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(ITensor *input, ITensor *output, const ActivationLayerInfo &activation_info)
{
    auto activation = arm_compute::support::cpp14::make_unique<ActivationType>();
    activation->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(output),
        activation_info);

    return std::move(activation);
}

template <Hint                          hint>
std::unique_ptr<arm_compute::IFunction> instantiate(ITensor *input, ITensor *output, const ActivationLayerInfo &activation_info);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<Hint::OPENCL>(ITensor *input, ITensor *output, const ActivationLayerInfo &activation_info)
{
    return instantiate_function<arm_compute::CLActivationLayer, arm_compute::CLTensor, Hint::OPENCL>(input, output, activation_info);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<Hint::NEON>(ITensor *input, ITensor *output, const ActivationLayerInfo &activation_info)
{
    return instantiate_function<arm_compute::NEActivationLayer, arm_compute::Tensor, Hint::NEON>(input, output, activation_info);
}
} // namespace

ActivationLayer::ActivationLayer(const ActivationLayerInfo activation_info)
    : _activation_info(activation_info)
{
}

std::unique_ptr<arm_compute::IFunction> ActivationLayer::instantiate_node(Hint hint, ITensor *input, ITensor *output)
{
    std::unique_ptr<arm_compute::IFunction> func;
    _hint   = hint;
    _input  = input;
    _output = output;

    if(_hint == Hint::OPENCL)
    {
        func = instantiate<Hint::OPENCL>(input, output, _activation_info);
    }
    else
    {
        func = instantiate<Hint::NEON>(input, output, _activation_info);
    }
    return func;
}

void ActivationLayer::print_info()
{
    if(_hint == Hint::OPENCL)
    {
        std::cout << "Instantiating CLActivationLayer";
    }
    else
    {
        std::cout << "Instantiating NEActivationLayer";
    }

    std::cout << " Data Type: " << _input->info()->data_type()
              << " Input shape: " << _input->info()->tensor_shape()
              << " Output shape: " << _output->info()->tensor_shape()
              << " Activation function: " << _activation_info.activation()
              << " a: " << _activation_info.a()
              << " b: " << _activation_info.b()
              << std::endl;
}
