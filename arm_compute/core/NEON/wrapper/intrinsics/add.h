/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_WRAPPER_ADD_H__
#define __ARM_COMPUTE_WRAPPER_ADD_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VADD_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vadd(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VADD_IMPL(uint8x8_t, uint8x8_t, vadd, u8)
VADD_IMPL(int8x8_t, int8x8_t, vadd, s8)
VADD_IMPL(uint16x4_t, uint16x4_t, vadd, u16)
VADD_IMPL(int16x4_t, int16x4_t, vadd, s16)
VADD_IMPL(uint32x2_t, uint32x2_t, vadd, u32)
VADD_IMPL(int32x2_t, int32x2_t, vadd, s32)
VADD_IMPL(uint64x1_t, uint64x1_t, vadd, u64)
VADD_IMPL(int64x1_t, int64x1_t, vadd, s64)
VADD_IMPL(float32x2_t, float32x2_t, vadd, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VADD_IMPL(float16x4_t, float16x4_t, vadd, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VADD_IMPL(uint8x16_t, uint8x16_t, vaddq, u8)
VADD_IMPL(int8x16_t, int8x16_t, vaddq, s8)
VADD_IMPL(uint16x8_t, uint16x8_t, vaddq, u16)
VADD_IMPL(int16x8_t, int16x8_t, vaddq, s16)
VADD_IMPL(uint32x4_t, uint32x4_t, vaddq, u32)
VADD_IMPL(int32x4_t, int32x4_t, vaddq, s32)
VADD_IMPL(uint64x2_t, uint64x2_t, vaddq, u64)
VADD_IMPL(int64x2_t, int64x2_t, vaddq, s64)
VADD_IMPL(float32x4_t, float32x4_t, vaddq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VADD_IMPL(float16x8_t, float16x8_t, vaddq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#undef VADD_IMPL

#define VQADD_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vqadd(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

// VQADD: Vector saturating add (No notion of saturation for floating point)
VQADD_IMPL(uint8x8_t, uint8x8_t, vqadd, u8)
VQADD_IMPL(int8x8_t, int8x8_t, vqadd, s8)
VQADD_IMPL(uint16x4_t, uint16x4_t, vqadd, u16)
VQADD_IMPL(int16x4_t, int16x4_t, vqadd, s16)
VQADD_IMPL(uint32x2_t, uint32x2_t, vqadd, u32)
VQADD_IMPL(int32x2_t, int32x2_t, vqadd, s32)
VQADD_IMPL(uint64x1_t, uint64x1_t, vqadd, u64)
VQADD_IMPL(int64x1_t, int64x1_t, vqadd, s64)
VQADD_IMPL(float32x2_t, float32x2_t, vadd, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VQADD_IMPL(float16x4_t, float16x4_t, vadd, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VQADD_IMPL(uint8x16_t, uint8x16_t, vqaddq, u8)
VQADD_IMPL(int8x16_t, int8x16_t, vqaddq, s8)
VQADD_IMPL(uint16x8_t, uint16x8_t, vqaddq, u16)
VQADD_IMPL(int16x8_t, int16x8_t, vqaddq, s16)
VQADD_IMPL(uint32x4_t, uint32x4_t, vqaddq, u32)
VQADD_IMPL(int32x4_t, int32x4_t, vqaddq, s32)
VQADD_IMPL(uint64x2_t, uint64x2_t, vqaddq, u64)
VQADD_IMPL(int64x2_t, int64x2_t, vqaddq, s64)
VQADD_IMPL(float32x4_t, float32x4_t, vaddq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VQADD_IMPL(float16x8_t, float16x8_t, vaddq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#undef VQADD_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_ADD_H__ */
