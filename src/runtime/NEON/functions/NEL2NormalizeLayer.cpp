/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
namespace
{
constexpr int max_input_tensor_dim = 3;
} // namespace

NEL2NormalizeLayer::NEL2NormalizeLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _reduce_func(), _normalize_kernel(), _sumsq()
{
}

void NEL2NormalizeLayer::configure(ITensor *input, ITensor *output, int axis, float epsilon)
{
    // Manage intermediate buffers
    _memory_group.manage(&_sumsq);

    // Configure Kernels
    const uint32_t actual_axis = wrap_around(axis, max_input_tensor_dim);
    _reduce_func.configure(input, &_sumsq, actual_axis, ReductionOperation::SUM_SQUARE);
    _normalize_kernel.configure(input, &_sumsq, output, axis, epsilon);

    // Allocate intermediate tensors
    _sumsq.allocator()->allocate();
}

Status NEL2NormalizeLayer::validate(const ITensorInfo *input, const ITensorInfo *output, int axis, float epsilon)
{
    TensorShape shape(input->tensor_shape());

    // Create intermediate tensor info
    TensorInfo sum_sq;
    sum_sq.set_data_type(input->data_type());
    sum_sq.set_tensor_shape(shape);

    const uint32_t actual_axis = wrap_around(axis, max_input_tensor_dim);
    ARM_COMPUTE_RETURN_ON_ERROR(NEReductionOperation::validate(input, &sum_sq, actual_axis, ReductionOperation::SUM_SQUARE));

    // Reduce shape on axis
    shape.set(actual_axis, 1);
    sum_sq.set_tensor_shape(shape);

    ARM_COMPUTE_RETURN_ON_ERROR(NEL2NormalizeLayerKernel::validate(input, &sum_sq, output, axis, epsilon));

    return Status{};
}

void NEL2NormalizeLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    _reduce_func.run();
    NEScheduler::get().schedule(&_normalize_kernel, Window::DimY);
}
} // namespace arm_compute
