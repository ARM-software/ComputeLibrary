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
#ifndef ACL_SRC_CPU_KERNELS_DYNAMIC_GEMM_GENERIC_IMPL_H
#define ACL_SRC_CPU_KERNELS_DYNAMIC_GEMM_GENERIC_IMPL_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{

#define DECLARE_DYNAMIC_GEMM_KERNEL(kernel_name)                                                                \
    void   kernel_name##_run(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, ITensor *pack_b, \
                             const Window &window);                                                             \
    void   kernel_name##_pack_rhs(const ITensor *rhs, const ITensor *bias, ITensor *pack_b);                    \
    size_t kernel_name##_size_of_packed_rhs(size_t rows, size_t columns);                                       \
    Window kernel_name##_window(const ITensorInfo *dst)

#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)
DECLARE_DYNAMIC_GEMM_KERNEL(neon_fp32_dynamic_gemm);
#endif // __aarch64__ && ENABLE_FP32_KERNELS

#undef DECLARE_DYNAMIC_GEMM_KERNEL

} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_DYNAMIC_GEMM_GENERIC_IMPL_H
