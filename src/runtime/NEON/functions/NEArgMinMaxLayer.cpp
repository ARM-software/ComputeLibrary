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

#include "arm_compute/runtime/NEON/functions/NEArgMinMaxLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
NEArgMinMaxLayer::NEArgMinMaxLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _reduction_kernel(), _fill_border_kernel(), _run_fill_border(false)
{
}
void NEArgMinMaxLayer::configure(ITensor *input, int axis, ITensor *output, const ReductionOperation &op)
{
    _reduction_kernel.configure(input, output, axis, op);

    if(axis == 0)
    {
        _fill_border_kernel.configure(input, _reduction_kernel.border_size(), BorderMode::REPLICATE);
        _run_fill_border = true;
    }
}

Status NEArgMinMaxLayer::validate(const ITensorInfo *input, int axis, const ITensorInfo *output, const ReductionOperation &op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(op != ReductionOperation::ARG_IDX_MAX && op != ReductionOperation::ARG_IDX_MIN, "Invalid operation");
    ARM_COMPUTE_RETURN_ON_ERROR(NEReductionOperationKernel::validate(input, output, axis, op));
    return Status{};
}

void NEArgMinMaxLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_run_fill_border)
    {
        NEScheduler::get().schedule(&_fill_border_kernel, Window::DimY);
    }
    NEScheduler::get().schedule(&_reduction_kernel, Window::DimY);
}

} // namespace arm_compute