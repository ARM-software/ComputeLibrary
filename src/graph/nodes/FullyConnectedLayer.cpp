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
#include "arm_compute/graph/nodes/FullyConnectedLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "support/ToolchainSupport.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename FullyConnectedType, typename TensorType, Hint hint>
std::unique_ptr<arm_compute::IFunction> instantiate_function(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output)
{
    bool weights_are_loaded = weights.tensor() != nullptr;
    bool biases_are_loaded  = biases.tensor() != nullptr;

    auto conv = arm_compute::support::cpp14::make_unique<FullyConnectedType>();
    conv->configure(
        dynamic_cast<TensorType *>(input),
        dynamic_cast<TensorType *>(weights.set_target(hint)),
        dynamic_cast<TensorType *>(biases.set_target(hint)),
        dynamic_cast<TensorType *>(output));
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
std::unique_ptr<arm_compute::IFunction> instantiate(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output);

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<Hint::OPENCL>(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output)
{
    return instantiate_function<arm_compute::CLFullyConnectedLayer, arm_compute::CLTensor, Hint::OPENCL>(input, weights, biases, output);
}

template <>
std::unique_ptr<arm_compute::IFunction> instantiate<Hint::NEON>(ITensor *input, Tensor &weights, Tensor &biases, ITensor *output)
{
    return instantiate_function<arm_compute::NEFullyConnectedLayer, arm_compute::Tensor, Hint::NEON>(input, weights, biases, output);
}
} // namespace

std::unique_ptr<arm_compute::IFunction> FullyConnectedLayer::instantiate_node(Hint hint, ITensor *input, ITensor *output)
{
    if(_weights.tensor() == nullptr)
    {
        unsigned int num_weights    = 1;
        unsigned int num_dimensions = input->info()->num_dimensions();
        // Ignore the batch dimension if there is one:
        if(num_dimensions == 2 || num_dimensions == 4)
        {
            num_dimensions--;
        }
        for(unsigned int i = 0; i < num_dimensions; i++)
        {
            num_weights *= input->info()->dimension(i);
        }
        _weights.set_info(TensorInfo(TensorShape(num_weights, _num_neurons), input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position()));
    }
    if(_biases.tensor() == nullptr)
    {
        _biases.set_info(TensorInfo(TensorShape(_num_neurons), input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position()));
    }

    arm_compute::auto_init_if_empty(*output->info(), TensorShape(_num_neurons, input->info()->dimension(1)), input->info()->num_channels(), input->info()->data_type(),
                                    input->info()->fixed_point_position());

    std::unique_ptr<arm_compute::IFunction> func;
    _hint   = hint;
    _input  = input;
    _output = output;

    if(_hint == Hint::OPENCL)
    {
        func = instantiate<Hint::OPENCL>(input, _weights, _biases, output);
    }
    else
    {
        func = instantiate<Hint::NEON>(input, _weights, _biases, output);
    }

    return func;
}

void FullyConnectedLayer::print_info()
{
    if(_hint == Hint::OPENCL)
    {
        std::cout << "Instantiating CLFullyConnectedLayer";
    }
    else
    {
        std::cout << "Instantiating NEFullyConnectedLayer";
    }
    std::cout << " Type: " << _input->info()->data_type() << " Input Shape: " << _input->info()->tensor_shape() << " Weights shape: " << _weights.info().tensor_shape() << " Biases Shape: " <<
              _biases.info().tensor_shape() << " Output Shape: " << _output->info()->tensor_shape() << std::endl;
}
