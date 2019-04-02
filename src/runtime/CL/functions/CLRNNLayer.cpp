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
#include "arm_compute/runtime/CL/functions/CLRNNLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

#include <utility>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLRNNLayer::CLRNNLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _gemm_state_f(), _add_kernel(), _activation_kernel(), _fully_connected_kernel(), _copy_kernel(), _fully_connected_out(), _gemm_output(), _add_output(),
      _is_prepared(false)
{
}

Status CLRNNLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *recurrent_weights, const ITensorInfo *bias, const ITensorInfo *hidden_state,
                            const ITensorInfo *output, const ActivationLayerInfo &info)
{
    const int idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, recurrent_weights, bias, hidden_state, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_width) != weights->dimension(idx_width));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_height) != recurrent_weights->dimension(idx_width));
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_weights->dimension(idx_width) != recurrent_weights->dimension(1));
    ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(bias->dimension(idx_width) != weights->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON(hidden_state->dimension(idx_width) != weights->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON(hidden_state->dimension(idx_height) != input->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), hidden_state->tensor_shape());

    auto shape_info = TensorInfo(compute_rnn_shape(recurrent_weights, hidden_state->dimension(idx_height)), 1, input->data_type());

    ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, weights, bias, &shape_info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(hidden_state, recurrent_weights, nullptr, &shape_info, 1.f, 0.f));
    ARM_COMPUTE_RETURN_ON_ERROR(CLSaturatedArithmeticOperationKernel::validate(ArithmeticOperation::ADD, &shape_info, &shape_info, &shape_info, ConvertPolicy::SATURATE));
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(&shape_info, &shape_info, info));

    return Status{};
}

void CLRNNLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *recurrent_weights, const ICLTensor *bias, ICLTensor *hidden_state, ICLTensor *output,
                           ActivationLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, recurrent_weights, bias, hidden_state, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLRNNLayer::validate(input->info(), weights->info(), recurrent_weights->info(), bias->info(), hidden_state->info(), output->info(), info));

    const int   idx_height = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);
    TensorShape shape      = compute_rnn_shape(recurrent_weights->info(), hidden_state->info()->dimension(idx_height));

    _is_prepared = false;

    _fully_connected_out.allocator()->init(TensorInfo(shape, 1, input->info()->data_type()));
    _gemm_output.allocator()->init(TensorInfo(shape, 1, input->info()->data_type()));

    // Manage intermediate buffers and configure
    _memory_group.manage(&_fully_connected_out);
    _fully_connected_kernel.configure(input, weights, bias, &_fully_connected_out);

    _memory_group.manage(&_gemm_output);
    _gemm_state_f.configure(hidden_state, recurrent_weights, nullptr, &_gemm_output, 1.f, 0.f);

    _add_output.allocator()->init(TensorInfo(shape, 1, input->info()->data_type()));
    _memory_group.manage(&_add_output);

    _add_kernel.configure(ArithmeticOperation::ADD, &_fully_connected_out, &_gemm_output, &_add_output, ConvertPolicy::SATURATE);

    _fully_connected_out.allocator()->allocate();
    _gemm_output.allocator()->allocate();

    _activation_kernel.configure(&_add_output, hidden_state, info);
    _add_output.allocator()->allocate();

    _copy_kernel.configure(hidden_state, output);
}

void CLRNNLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    _fully_connected_kernel.run();
    _gemm_state_f.run();
    CLScheduler::get().enqueue(_add_kernel);
    CLScheduler::get().enqueue(_activation_kernel);

    // copy hidden out to output
    CLScheduler::get().enqueue(_copy_kernel);
}

void CLRNNLayer::prepare()
{
    if(!_is_prepared)
    {
        _fully_connected_kernel.prepare();
        _gemm_state_f.prepare();

        _is_prepared = true;
    }
}
