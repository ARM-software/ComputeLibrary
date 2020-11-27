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
#ifndef ARM_COMPUTE_WRAPPER_ABS_H
#define ARM_COMPUTE_WRAPPER_ABS_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VABS_IMPL(stype, vtype, prefix, postfix) \
    inline vtype vabs(const vtype &a)            \
    {                                            \
        return prefix##_##postfix(a);            \
    }

#define VQABS_IMPL(stype, vtype, prefix, postfix) \
    inline vtype vqabs(const vtype &a)            \
    {                                             \
        return prefix##_##postfix(a);             \
    }

// Absolute: vabs{q}_<type>. Vd[i] = |Va[i]|
VABS_IMPL(int8x8_t, int8x8_t, vabs, s8)
VABS_IMPL(int16x4_t, int16x4_t, vabs, s16)
VABS_IMPL(int32x2_t, int32x2_t, vabs, s32)
VABS_IMPL(float32x2_t, float32x2_t, vabs, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VABS_IMPL(float16x4_t, float16x4_t, vabs, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VABS_IMPL(int8x16_t, int8x16_t, vabsq, s8)
VABS_IMPL(int16x8_t, int16x8_t, vabsq, s16)
VABS_IMPL(int32x4_t, int32x4_t, vabsq, s32)
VABS_IMPL(float32x4_t, float32x4_t, vabsq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VABS_IMPL(float16x8_t, float16x8_t, vabsq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

// Saturating absolute: vqabs{q}_<type>. Vd[i] = sat(|Va[i]|)
VQABS_IMPL(int8x8_t, int8x8_t, vqabs, s8)
VQABS_IMPL(int16x4_t, int16x4_t, vqabs, s16)
VQABS_IMPL(int32x2_t, int32x2_t, vqabs, s32)

VQABS_IMPL(int8x16_t, int8x16_t, vqabsq, s8)
VQABS_IMPL(int16x8_t, int16x8_t, vqabsq, s16)
VQABS_IMPL(int32x4_t, int32x4_t, vqabsq, s32)

#undef VABS_IMPL
#undef VQABS_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_ABS_H */
