/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_MUL_H
#define ARM_COMPUTE_WRAPPER_MUL_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VMUL_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vmul(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VMUL_IMPL(uint8x8_t, uint8x8_t, vmul, u8)
VMUL_IMPL(int8x8_t, int8x8_t, vmul, s8)
VMUL_IMPL(uint16x4_t, uint16x4_t, vmul, u16)
VMUL_IMPL(int16x4_t, int16x4_t, vmul, s16)
VMUL_IMPL(uint32x2_t, uint32x2_t, vmul, u32)
VMUL_IMPL(int32x2_t, int32x2_t, vmul, s32)
VMUL_IMPL(float32x2_t, float32x2_t, vmul, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMUL_IMPL(float16_t, float16x4_t, vmul, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VMUL_IMPL(uint8_t, uint8x16_t, vmulq, u8)
VMUL_IMPL(int8_t, int8x16_t, vmulq, s8)
VMUL_IMPL(uint16_t, uint16x8_t, vmulq, u16)
VMUL_IMPL(int16_t, int16x8_t, vmulq, s16)
VMUL_IMPL(uint32_t, uint32x4_t, vmulq, u32)
VMUL_IMPL(int32_t, int32x4_t, vmulq, s32)
VMUL_IMPL(float32x4_t, float32x4_t, vmulq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMUL_IMPL(float16_t, float16x8_t, vmulq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VMUL_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_MUL_H */
