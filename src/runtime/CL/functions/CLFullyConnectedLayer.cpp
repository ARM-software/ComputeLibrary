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
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLFullyConnectedLayer::CLFullyConnectedLayer()
    : _conv_function(), _gemm_function(), _transpose_kernel(), _acc_biases_kernel(), _run_func(), _weights_transpose(), _is_first_run(true), _run_acc_biases(false)
{
}

void CLFullyConnectedLayer::configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *biases, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON((weights->info()->num_dimensions() != 2) && (weights->info()->num_dimensions() != 4));

    // Make sure that in the fully connected layer connected to fully connected layer case, the first dimension of the weights and input are same.
    ARM_COMPUTE_ERROR_ON((weights->info()->num_dimensions() == 2) && (input->info()->dimension(0) != weights->info()->dimension(0)));

    if(weights->info()->num_dimensions() != 2)
    {
        _conv_function.configure(input, weights, biases, output, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::FLOOR));
        _run_func = &CLFullyConnectedLayer::run_conv;
        return;
    }

    TensorShape shape_trans(weights->info()->dimension(1), weights->info()->dimension(0));
    _weights_transpose.allocator()->init(TensorInfo(shape_trans, 1, weights->info()->data_type()));

    // Configure kernels
    _transpose_kernel.configure(weights, &_weights_transpose);
    _gemm_function.configure(input, &_weights_transpose, nullptr, output, 1.0f, 0.0f);
    if(biases != nullptr)
    {
        _acc_biases_kernel.configure(output, biases);
        _run_acc_biases = true;
    }

    _run_func = &CLFullyConnectedLayer::run_fc;

    // Allocate intermediate buffers
    _weights_transpose.allocator()->allocate();
}

void CLFullyConnectedLayer::run_conv()
{
    _conv_function.run();
}

void CLFullyConnectedLayer::run_fc()
{
    if(_is_first_run)
    {
        _is_first_run = false;
        CLScheduler::get().enqueue(_transpose_kernel);
    }

    _gemm_function.run();

    if(_run_acc_biases)
    {
        CLScheduler::get().enqueue(_acc_biases_kernel);
    }
}

void CLFullyConnectedLayer::run()
{
    ARM_COMPUTE_ERROR_ON(_run_func == nullptr);
    (this->*_run_func)();
}
