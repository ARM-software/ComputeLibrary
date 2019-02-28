/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_WRAPPER_MLA_H__
#define __ARM_COMPUTE_WRAPPER_MLA_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VMLA_IMPL(stype, vtype, prefix, postfix)                      \
    inline vtype vmla(const vtype &a, const vtype &b, const vtype &c) \
    {                                                                 \
        return prefix##_##postfix(a, b, c);                           \
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define VMLA_IMPL2(stype, vtype, prefix1, prefix2, postfix)           \
    inline vtype vmla(const vtype &a, const vtype &b, const vtype &c) \
    {                                                                 \
        return prefix1##_##postfix(a, prefix2##_##postfix(b, c));     \
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VMLA_IMPL(uint8x8_t, uint8x8_t, vmla, u8)
VMLA_IMPL(int8x8_t, int8x8_t, vmla, s8)
VMLA_IMPL(uint16x4_t, uint16x4_t, vmla, u16)
VMLA_IMPL(int16x4_t, int16x4_t, vmla, s16)
VMLA_IMPL(uint32x2_t, uint32x2_t, vmla, u32)
VMLA_IMPL(int32x2_t, int32x2_t, vmla, s32)
VMLA_IMPL(float32x2_t, float32x2_t, vmla, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMLA_IMPL2(float16x4_t, float16x4_t, vadd, vmul, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VMLA_IMPL(uint8x16_t, uint8x16_t, vmlaq, u8)
VMLA_IMPL(int8x16_t, int8x16_t, vmlaq, s8)
VMLA_IMPL(uint16x8_t, uint16x8_t, vmlaq, u16)
VMLA_IMPL(int16x8_t, int16x8_t, vmlaq, s16)
VMLA_IMPL(uint32x4_t, uint32x4_t, vmlaq, u32)
VMLA_IMPL(int32x4_t, int32x4_t, vmlaq, s32)
VMLA_IMPL(float32x4_t, float32x4_t, vmlaq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMLA_IMPL2(float16x8_t, float16x8_t, vaddq, vmulq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VMLA_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_MLA_H__ */
