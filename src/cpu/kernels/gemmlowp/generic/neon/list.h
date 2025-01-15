/*
 * Copyright (c) 2025 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_GEMMLOWP_GENERIC_NEON_LIST_H
#define ACL_SRC_CPU_KERNELS_GEMMLOWP_GENERIC_NEON_LIST_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

#include <cstdint>

namespace arm_compute
{
namespace cpu
{

#define DECLARE_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL(func_name)                                                       \
    void func_name(const Window &window, ITensor *mm_result, const ITensor *vector_sum_col,                          \
                   const ITensor *vector_sum_row, int32_t a_offset, int32_t b_offset, int32_t k_offset, float scale, \
                   bool slide_vector_sum_col, bool is_gemm3d)

DECLARE_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL(neon_run_offset_contribution_fp32);
DECLARE_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL(neon_run_offset_contribution_fp16);
DECLARE_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL(neon_run_offset_contribution_int32);

#undef DECLARE_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_GEMMLOWP_GENERIC_NEON_LIST_H
