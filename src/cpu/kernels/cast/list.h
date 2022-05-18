/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_CAST_LIST_H
#define SRC_CORE_NEON_KERNELS_CAST_LIST_H
namespace arm_compute
{
namespace cpu
{
#define DECLARE_CAST_KERNEL(func_name) \
    void func_name(const ITensor *_src, ITensor *_dst, const ThreadInfo &tensor, ConvertPolicy _policy, const Window &window)

DECLARE_CAST_KERNEL(neon_fp32_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_u8_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_fp16_to_other_dt_cast);
DECLARE_CAST_KERNEL(neon_s32_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_qasymm8_signed_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_fp32_to_bfloat16_cast);
DECLARE_CAST_KERNEL(neon_bfloat16_to_fp32_cast);

#undef DECLARE_CAST_KERNEL
} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_CAST_LIST_H