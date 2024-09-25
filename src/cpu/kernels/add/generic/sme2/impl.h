/*
 * Copyright (c) 2024 Arm Limited.
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

#ifndef ACL_SRC_CPU_KERNELS_ADD_GENERIC_SME2_IMPL_H
#define ACL_SRC_CPU_KERNELS_ADD_GENERIC_SME2_IMPL_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h" // Needed for ConvertPolicy
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{

void add_qasymm8_signed_sme2(
    const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window);

bool add_q8_sme2_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

bool add_sub_q8_sme2_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_ADD_GENERIC_SME2_IMPL_H
