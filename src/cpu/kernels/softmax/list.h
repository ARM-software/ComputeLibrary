/*
 * Copyright (c) 2021-2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_SOFTMAX_LIST_H
#define ACL_SRC_CPU_KERNELS_SOFTMAX_LIST_H

namespace arm_compute
{
namespace cpu
{
#define DECLARE_SOFTMAX_KERNEL(func_name)                                                                              \
    template <bool IS_LOG>                                                                                             \
    void func_name(const ITensor *in, void *const tmp, ITensor *out, const float beta, int axis, const Window &window, \
                   const void *lut_ptr)

DECLARE_SOFTMAX_KERNEL(neon_fp32_softmax);
DECLARE_SOFTMAX_KERNEL(neon_fp16_softmax);
DECLARE_SOFTMAX_KERNEL(neon_qasymm8_softmax);
DECLARE_SOFTMAX_KERNEL(neon_qasymm8_signed_softmax);

#ifdef ARM_COMPUTE_ENABLE_SME2

void sme2_fp32_softmax(const ITensor *in,
                       void *const    tmp,
                       ITensor       *out,
                       const float    beta,
                       int            axis,
                       const Window  &window,
                       const void    *lut_ptr);

void sme2_fp16_softmax(const ITensor *in,
                       void *const    tmp,
                       ITensor       *out,
                       const float    beta,
                       int            axis,
                       const Window  &window,
                       const void    *lut_ptr);

void sme2_qasymm8_softmax_lut_512VL(const ITensor *in,
                                    void *const    tmp,
                                    ITensor       *out,
                                    const float    beta,
                                    int            axis,
                                    const Window  &window,
                                    const void    *lut_ptr);

void sme2_qasymm8_signed_softmax_lut_512VL(const ITensor *in,
                                           void *const    tmp,
                                           ITensor       *out,
                                           const float    beta,
                                           int            axis,
                                           const Window  &window,
                                           const void    *lut_ptr);

#endif // ARM_COMPUTE_ENABLE_SME2

#ifdef ARM_COMPUTE_ENABLE_BF16

void sve_softmax_bf16(const ITensor *in,
                      void *const    tmp,
                      ITensor       *out,
                      const float    beta,
                      int            axis,
                      const Window  &window,
                      const void    *lut_ptr);

#endif // ARM_COMPUTE_ENABLE_BF16

#undef DECLARE_SOFTMAX_KERNEL
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_SOFTMAX_LIST_H
