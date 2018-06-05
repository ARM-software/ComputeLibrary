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

#include "arm_compute/runtime/NEON/functions/NERNNLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
NERNNLayer::NERNNLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _gemm_state_f(), _add_kernel(), _activation_kernel(), _fully_connected_kernel(), _fully_connected_out(), _gemm_output(), _add_output(), _hidden_state(),
      _output()
{
}

Status NERNNLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *recurrent_weights, const ITensorInfo *bias, const ITensorInfo *hidden_state,
                            const ITensorInfo *output, const ActivationLayerInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, recurrent_weights, bias, hidden_state, output);

    const int idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_width) != weights->dimension(idx_width));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_height) != recurrent_weights->dimension(idx_width));
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_weights->dimension(idx_width) != recurrent_weights->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(bias->dimension(idx_width) != weights->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON(hidden_state->dimension(idx_width) != weights->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON(hidden_state->dimension(idx_height) != input->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), hidden_state->tensor_shape());

    auto shape_info = TensorInfo(misc::shape_calculator::compute_rnn_shape(recurrent_weights, hidden_state->dimension(idx_height)), 1, input->data_type());

    ARM_COMPUTE_RETURN_ON_ERROR(NEFullyConnectedLayer::validate(input, weights, bias, &shape_info, true, false));
    ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAdditionKernel::validate(&shape_info, &shape_info, &shape_info, ConvertPolicy::SATURATE));
    ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayerKernel::validate(&shape_info, &shape_info, info));

    return Status{};
}

void NERNNLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *recurrent_weights, const ITensor *bias, ITensor *hidden_state, ITensor *output,
                           ActivationLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, recurrent_weights, bias, hidden_state, output);
    ARM_COMPUTE_ERROR_THROW_ON(NERNNLayer::validate(input->info(), weights->info(), recurrent_weights->info(), bias->info(), hidden_state->info(), output->info(), info));

    _hidden_state = hidden_state;
    _output       = output;

    const int   idx_height = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);
    TensorShape shape      = misc::shape_calculator::compute_rnn_shape(recurrent_weights->info(), hidden_state->info()->dimension(idx_height));

    // Manage intermediate buffers and configure
    _fully_connected_out.allocator()->init(TensorInfo(shape, 1, input->info()->data_type()));
    _memory_group.manage(&_fully_connected_out);
    _fully_connected_kernel.configure(input, weights, bias, &_fully_connected_out, true, false);

    _gemm_output.allocator()->init(TensorInfo(shape, 1, input->info()->data_type()));
    _memory_group.manage(&_gemm_output);
    _gemm_state_f.configure(hidden_state, recurrent_weights, nullptr, &_gemm_output, 1.f, 0.f);

    _add_output.allocator()->init(TensorInfo(shape, 1, input->info()->data_type()));
    _memory_group.manage(&_add_output);
    _add_kernel.configure(&_fully_connected_out, &_gemm_output, &_add_output, ConvertPolicy::SATURATE);

    _fully_connected_out.allocator()->allocate();
    _gemm_output.allocator()->allocate();

    _activation_kernel.configure(&_add_output, hidden_state, info);
    _add_output.allocator()->allocate();
}

void NERNNLayer::run()
{
    _memory_group.acquire();

    _fully_connected_kernel.run();
    _gemm_state_f.run();
    NEScheduler::get().schedule(&_add_kernel, Window::DimY);
    NEScheduler::get().schedule(&_activation_kernel, Window::DimY);

    // copy hidden out to output
    Window hidden_state_window;
    Window output_window;
    hidden_state_window.use_tensor_dimensions(_hidden_state->info()->tensor_shape(), Window::DimY);
    output_window.use_tensor_dimensions(_output->info()->tensor_shape(), Window::DimY);

    Iterator hidden_state_it(_hidden_state, output_window);
    Iterator output_it(_output, output_window);

    execute_window_loop(output_window, [&](const Coordinates & id)
    {
        memcpy(output_it.ptr(), hidden_state_it.ptr(), _output->info()->dimension(0) * _output->info()->element_size());
    },
    hidden_state_it, output_it);

    _memory_group.release();
}
} // namespace arm_compute
