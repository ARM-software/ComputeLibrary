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
#ifndef ARM_COMPUTE_WRAPPER_ADD_H
#define ARM_COMPUTE_WRAPPER_ADD_H

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

// VQADD: Vector saturating add (No notion of saturation for floating point)
#define VQADD_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vqadd(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

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

// VADDW: Vector widening add
#define VADDW_IMPL(wtype, vtype, prefix, postfix)      \
    inline wtype vaddw(const wtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

VADDW_IMPL(uint16x8_t, uint8x8_t, vaddw, u8)
VADDW_IMPL(int16x8_t, int8x8_t, vaddw, s8)
VADDW_IMPL(uint32x4_t, uint16x4_t, vaddw, u16)
VADDW_IMPL(int32x4_t, int16x4_t, vaddw, s16)
VADDW_IMPL(uint64x2_t, uint32x2_t, vaddw, u32)
VADDW_IMPL(int64x2_t, int32x2_t, vaddw, s32)
#undef VADDW_IMPL

// VADDL: Vector long add
#define VADDL_IMPL(wtype, vtype, prefix, postfix)      \
    inline wtype vaddl(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

VADDL_IMPL(uint16x8_t, uint8x8_t, vaddl, u8)
VADDL_IMPL(int16x8_t, int8x8_t, vaddl, s8)
VADDL_IMPL(uint32x4_t, uint16x4_t, vaddl, u16)
VADDL_IMPL(int32x4_t, int16x4_t, vaddl, s16)
VADDL_IMPL(uint64x2_t, uint32x2_t, vaddl, u32)
VADDL_IMPL(int64x2_t, int32x2_t, vaddl, s32)
#undef VADDL_IMPL

#if defined(__aarch64__)
// VADDV: Across vector add
#define VADDV_IMPL(stype, vtype, prefix, postfix) \
    inline stype vaddv(const vtype &a)            \
    {                                             \
        return prefix##_##postfix(a);             \
    }

VADDV_IMPL(uint8_t, uint8x8_t, vaddv, u8)
VADDV_IMPL(int8_t, int8x8_t, vaddv, s8)
VADDV_IMPL(uint16_t, uint16x4_t, vaddv, u16)
VADDV_IMPL(int16_t, int16x4_t, vaddv, s16)
VADDV_IMPL(uint32_t, uint32x2_t, vaddv, u32)
VADDV_IMPL(int32_t, int32x2_t, vaddv, s32)
VADDV_IMPL(float, float32x2_t, vaddv, f32)

VADDV_IMPL(uint8_t, uint8x16_t, vaddvq, u8)
VADDV_IMPL(int8_t, int8x16_t, vaddvq, s8)
VADDV_IMPL(uint16_t, uint16x8_t, vaddvq, u16)
VADDV_IMPL(int16_t, int16x8_t, vaddvq, s16)
VADDV_IMPL(uint32_t, uint32x4_t, vaddvq, u32)
VADDV_IMPL(int32_t, int32x4_t, vaddvq, s32)
VADDV_IMPL(uint64_t, uint64x2_t, vaddvq, u64)
VADDV_IMPL(int64_t, int64x2_t, vaddvq, s64)
VADDV_IMPL(float, float32x4_t, vaddvq, f32)
#undef VADDV_IMPL
#endif // defined(__aarch64__)

// VPADDL: Signed add long pairwise
#define VPADDL_IMPL(ltype, vtype, prefix, postfix) \
    inline ltype vpaddl(const vtype &a)            \
    {                                              \
        return prefix##_##postfix(a);              \
    }

VPADDL_IMPL(uint16x4_t, uint8x8_t, vpaddl, u8)
VPADDL_IMPL(int16x4_t, int8x8_t, vpaddl, s8)
VPADDL_IMPL(uint32x2_t, uint16x4_t, vpaddl, u16)
VPADDL_IMPL(int32x2_t, int16x4_t, vpaddl, s16)
VPADDL_IMPL(uint64x1_t, uint32x2_t, vpaddl, u32)
VPADDL_IMPL(int64x1_t, int32x2_t, vpaddl, s32)

VPADDL_IMPL(uint16x8_t, uint8x16_t, vpaddlq, u8)
VPADDL_IMPL(int16x8_t, int8x16_t, vpaddlq, s8)
VPADDL_IMPL(uint32x4_t, uint16x8_t, vpaddlq, u16)
VPADDL_IMPL(int32x4_t, int16x8_t, vpaddlq, s16)
VPADDL_IMPL(uint64x2_t, uint32x4_t, vpaddlq, u32)
VPADDL_IMPL(int64x2_t, int32x4_t, vpaddlq, s32)
#undef VPADDL_IMPL

// VPADD: Add pairwise
#define VPADD_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vpadd(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

VPADD_IMPL(uint8x8_t, uint8x8_t, vpadd, u8)
VPADD_IMPL(int8x8_t, int8x8_t, vpadd, s8)
VPADD_IMPL(uint16x4_t, uint16x4_t, vpadd, u16)
VPADD_IMPL(int16x4_t, int16x4_t, vpadd, s16)
VPADD_IMPL(uint32x2_t, uint32x2_t, vpadd, u32)
VPADD_IMPL(int32x2_t, int32x2_t, vpadd, s32)
VPADD_IMPL(float32x2_t, float32x2_t, vpadd, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VPADD_IMPL(float16x4_t, float16x4_t, vpadd, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VPADD_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_ADD_H */
