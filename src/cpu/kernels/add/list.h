/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef SRC_CORE_KERNELS_ADD_LIST_H
#define SRC_CORE_KERNELS_ADD_LIST_H

#include "src/cpu/kernels/add/generic/neon/impl.h"
#include "src/cpu/kernels/add/generic/sve/impl.h"

namespace arm_compute
{
namespace cpu
{
#define DECLARE_ADD_KERNEL(func_name) \
    void func_name(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)

DECLARE_ADD_KERNEL(add_qasymm8_neon);
DECLARE_ADD_KERNEL(add_qasymm8_signed_neon);
DECLARE_ADD_KERNEL(add_qsymm16_neon);
DECLARE_ADD_KERNEL(add_fp32_neon);
DECLARE_ADD_KERNEL(add_fp16_neon);
DECLARE_ADD_KERNEL(add_u8_neon);
DECLARE_ADD_KERNEL(add_s16_neon);
DECLARE_ADD_KERNEL(add_s32_neon);
DECLARE_ADD_KERNEL(add_fp32_sve);
DECLARE_ADD_KERNEL(add_fp16_sve);
DECLARE_ADD_KERNEL(add_u8_sve);
DECLARE_ADD_KERNEL(add_s16_sve);
DECLARE_ADD_KERNEL(add_s32_sve);
DECLARE_ADD_KERNEL(add_qasymm8_sve2);
DECLARE_ADD_KERNEL(add_qasymm8_signed_sve2);
DECLARE_ADD_KERNEL(add_qsymm16_sve2);

#undef DECLARE_ADD_KERNEL

} // namespace cpu
} // namespace arm_compute
#endif // SRC_CORE_KERNELS_ADD_LIST_H