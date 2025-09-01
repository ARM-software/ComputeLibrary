/*
 * Copyright (c) 2018-2021, 2023-2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEReverse.h"

#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/core/NEON/kernels/NEReverseKernel.h"

namespace arm_compute
{
void NEReverse::configure(const ITensor *input, ITensor *output, const ITensor *axis, bool use_inverted_axis)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEReverse::configure");
    ARM_COMPUTE_LOG_PARAMS(input, output, axis);

    auto k = std::make_unique<NEReverseKernel>();
    k->configure(input, output, axis, use_inverted_axis);
    _kernel = std::move(k);
}

Status NEReverse::validate(const ITensorInfo *input,
                           const ITensorInfo *output,
                           const ITensorInfo *axis,
                           bool               use_inverted_axis)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEReverse::validate");
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, output, axis);
    return NEReverseKernel::validate(input, output, axis, use_inverted_axis);
}
} // namespace arm_compute
