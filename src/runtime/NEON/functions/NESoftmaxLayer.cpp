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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cfloat>

using namespace arm_compute;

NESoftmaxLayer::NESoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _max_kernel(), _shift_exp_sum_kernel(), _norm_kernel(), _fill_border_kernel(), _max(), _sum(), _tmp()
{
}

void NESoftmaxLayer::configure(ITensor *input, ITensor *output, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Create intermediate tensors shapes
    TensorInfo tensor_info_tmp(input->info()->tensor_shape(), input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position());
    _tmp.allocator()->init(tensor_info_tmp);

    TensorShape shape = input->info()->tensor_shape();
    shape.set(0, 1);
    TensorInfo tensor_info_max_sum(shape, input->info()->num_channels(), input->info()->data_type(), input->info()->fixed_point_position());
    _max.allocator()->init(tensor_info_max_sum);
    _sum.allocator()->init(tensor_info_max_sum);

    // Manage intermediate buffers
    _memory_group.manage(&_tmp);
    _memory_group.manage(&_max);
    _memory_group.manage(&_sum);

    // Configure Kernels
    _max_kernel.configure(input, &_max);
    _shift_exp_sum_kernel.configure(input, &_max, &_tmp, &_sum, beta);
    _norm_kernel.configure(&_tmp, &_sum, output);
    _fill_border_kernel.configure(input, _max_kernel.border_size(), BorderMode::REPLICATE);

    // Allocate intermediate tensors
    _tmp.allocator()->allocate();
    _max.allocator()->allocate();
    _sum.allocator()->allocate();
}

Status NESoftmaxLayer::validate(const ITensorInfo *input, const ITensorInfo *output, float beta)
{
    // Perform validation step
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);

    TensorInfo tensor_info_max_sum(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(max_sum_shape));

    ARM_COMPUTE_RETURN_ON_ERROR(NELogits1DMaxKernel::validate(input, &tensor_info_max_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(NELogits1DShiftExpSumKernel::validate(input, &tensor_info_max_sum, input, &tensor_info_max_sum, beta));
    ARM_COMPUTE_RETURN_ON_ERROR(NELogits1DNormKernel::validate(input, &tensor_info_max_sum, output));

    return Status{};
}

void NESoftmaxLayer::run()
{
    _memory_group.acquire();

    NEScheduler::get().schedule(&_fill_border_kernel, Window::DimY);
    NEScheduler::get().schedule(&_max_kernel, Window::DimY);
    NEScheduler::get().schedule(&_shift_exp_sum_kernel, Window::DimY);
    NEScheduler::get().schedule(&_norm_kernel, Window::DimY);

    _memory_group.release();
}
