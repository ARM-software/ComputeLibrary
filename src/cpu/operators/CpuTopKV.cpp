/*
 * Copyright (c) 2026 Arm Limited.
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
#include "src/cpu/operators/CpuTopKV.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/cpu/kernels/CpuTopKVKernel.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
void CpuTopKV::configure(const ITensorInfo *predictions,
                         const ITensorInfo *targets,
                         ITensorInfo       *output,
                         const unsigned int k)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTopKV::configure");
    ARM_COMPUTE_LOG_PARAMS(predictions, targets, output, k);

    auto kernel = std::make_unique<kernels::CpuTopKVKernel>();
    kernel->configure(predictions, targets, output, k);
    _kernel = std::move(kernel);
}

Status CpuTopKV::validate(const ITensorInfo *predictions,
                          const ITensorInfo *targets,
                          ITensorInfo       *output,
                          const unsigned int k)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTopKV::validate");
    return kernels::CpuTopKVKernel::validate(predictions, targets, output, k);
}

void CpuTopKV::run(ITensorPack &tensors)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTopKV::run");
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

    NEScheduler::get().schedule_op(_kernel.get(), Window::DimX, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute
