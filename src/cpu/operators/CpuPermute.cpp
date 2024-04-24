/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/cpu/operators/CpuPermute.h"

#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuPermuteKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuPermute::configure(const ITensorInfo *src, ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, perm);
    auto k = std::make_unique<kernels::CpuPermuteKernel>();
    k->configure(src, dst, perm);
    _kernel = std::move(k);
}

Status CpuPermute::validate(const ITensorInfo *src, const ITensorInfo *dst, const PermutationVector &perm)
{
    return kernels::CpuPermuteKernel::validate(src, dst, perm);
}
} // namespace cpu
} // namespace arm_compute
