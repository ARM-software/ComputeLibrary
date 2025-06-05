/*
 * Copyright (c) 2018-2021, 2024-2025 Arm Limited.
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

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensorInfo.h"

#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/cpu/kernels/CpuCopyKernel.h"
#include "src/cpu/kernels/CpuPermuteKernel.h"
#include "src/cpu/kernels/CpuTransposeKernel.h"

#include <algorithm>
#include <array>
#include <memory>

namespace arm_compute
{
namespace cpu
{
namespace
{
// Handle "No-op" cases
bool prefer_copy(const PermutationVector &v)
{
    static const std::array<PermutationVector, 6> permutations = {{
        PermutationVector(0U),
        PermutationVector(0U, 1U),
        PermutationVector(0U, 1U, 2U),
        PermutationVector(0U, 1U, 2U, 3U),
        PermutationVector(0U, 1U, 2U, 3U, 4U),
        PermutationVector(0U, 1U, 2U, 3U, 4U, 5U),
    }};

    return std::find(permutations.begin(), permutations.end(), v) != permutations.end();
}

// Transpose kernel is optimized for permuting the first two dimensions of a tensor
bool prefer_transpose(const PermutationVector &v)
{
    static const std::array<PermutationVector, 5> permutations = {{
        PermutationVector(1U, 0U),
        PermutationVector(1U, 0U, 2U),
        PermutationVector(1U, 0U, 2U, 3U),
        PermutationVector(1U, 0U, 2U, 3U, 4U),
        PermutationVector(1U, 0U, 2U, 3U, 4U, 5U),
    }};

    return std::find(permutations.begin(), permutations.end(), v) != permutations.end();
}
} // namespace

void CpuPermute::configure(const ITensorInfo *src, ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuPermute::configure");
    ARM_COMPUTE_LOG_PARAMS(src, dst, perm);

    if (prefer_copy(perm))
    {
        auto k = std::make_unique<kernels::CpuCopyKernel>();
        k->configure(src, dst);
        _kernel = std::move(k);
    }
    else if (prefer_transpose(perm))
    {
        auto k = std::make_unique<kernels::CpuTransposeKernel>();
        k->configure(src, dst);
        _kernel = std::move(k);
    }
    else
    {
        auto k = std::make_unique<kernels::CpuPermuteKernel>();
        k->configure(src, dst, perm);
        _kernel = std::move(k);
    }
}

Status CpuPermute::validate(const ITensorInfo *src, const ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuPermute::validate");
    if (prefer_copy(perm))
    {
        return kernels::CpuCopyKernel::validate(src, dst);
    }

    if (prefer_transpose(perm))
    {
        return kernels::CpuTransposeKernel::validate(src, dst);
    }

    return kernels::CpuPermuteKernel::validate(src, dst, perm);
}
} // namespace cpu
} // namespace arm_compute
