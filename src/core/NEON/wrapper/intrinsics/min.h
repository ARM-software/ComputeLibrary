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
#ifndef ARM_COMPUTE_WRAPPER_MIN_H
#define ARM_COMPUTE_WRAPPER_MIN_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VMIN_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vmin(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VMIN_IMPL(uint8_t, uint8x8_t, vmin, u8)
VMIN_IMPL(int8_t, int8x8_t, vmin, s8)
VMIN_IMPL(uint16_t, uint16x4_t, vmin, u16)
VMIN_IMPL(int16_t, int16x4_t, vmin, s16)
VMIN_IMPL(uint32_t, uint32x2_t, vmin, u32)
VMIN_IMPL(int32_t, int32x2_t, vmin, s32)
VMIN_IMPL(float, float32x2_t, vmin, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMIN_IMPL(float16_t, float16x4_t, vmin, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VMIN_IMPL(uint8_t, uint8x16_t, vminq, u8)
VMIN_IMPL(int8_t, int8x16_t, vminq, s8)
VMIN_IMPL(uint16_t, uint16x8_t, vminq, u16)
VMIN_IMPL(int16_t, int16x8_t, vminq, s16)
VMIN_IMPL(uint32_t, uint32x4_t, vminq, u32)
VMIN_IMPL(int32_t, int32x4_t, vminq, s32)
VMIN_IMPL(float, float32x4_t, vminq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMIN_IMPL(float16_t, float16x8_t, vminq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VMIN_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_MIN_H */
