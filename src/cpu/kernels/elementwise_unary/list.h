/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_CORE_KERNELS_ELEMETWISE_UNARY_LIST_H
#define SRC_CORE_KERNELS_ELEMETWISE_UNARY_LIST_H

#include "src/cpu/kernels/elementwise_unary/generic/neon/impl.h"
#include "src/cpu/kernels/elementwise_unary/generic/sve/impl.h"

namespace arm_compute
{
namespace cpu
{
#define DECLARE_ELEMETWISE_UNARY_KERNEL(func_name) \
    void func_name(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op, const uint8_t *lut)

DECLARE_ELEMETWISE_UNARY_KERNEL(sve_fp32_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(sve_fp16_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(sve_s32_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(sve2_q8_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(neon_fp32_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(neon_fp16_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(neon_s32_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(neon_q8_elementwise_unary);
#ifndef __aarch64__
DECLARE_ELEMETWISE_UNARY_KERNEL(neon_qasymm8_signed_elementwise_unary);
DECLARE_ELEMETWISE_UNARY_KERNEL(neon_qasymm8_elementwise_unary);
#endif // __aarch64__
#undef DECLARE_ELEMETWISE_UNARY_KERNEL

} // namespace cpu
} // namespace arm_compute
#endif // SRC_CORE_KERNELS_ELEMETWISE_UNARY_LIST_H
