/*
 * Copyright (c) 2018-2021, 2025 Arm Limited.
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
#include "src/cpu/operators/CpuTranspose.h"

#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/cpu/kernels/CpuTransposeKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuTranspose::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTranspose::configure");
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    auto k = std::make_unique<kernels::CpuTransposeKernel>();
    k->configure(src, dst);
    _kernel = std::move(k);
}

Status CpuTranspose::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTranspose::validate");
    return kernels::CpuTransposeKernel::validate(src, dst);
}
} // namespace cpu
} // namespace arm_compute
