/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_CLE_H
#define ARM_COMPUTE_WRAPPER_CLE_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VCLE_IMPL(stype, vtype, rtype, prefix, postfix) \
    inline rtype vcle(const vtype &a, const vtype &b)   \
    {                                                   \
        return prefix##_##postfix(a, b);                \
    }

VCLE_IMPL(uint8_t, uint8x8_t, uint8x8_t, vcle, u8)
VCLE_IMPL(int8_t, int8x8_t, uint8x8_t, vcle, s8)
VCLE_IMPL(uint16_t, uint16x4_t, uint16x4_t, vcle, u16)
VCLE_IMPL(int16_t, int16x4_t, uint16x4_t, vcle, s16)
VCLE_IMPL(uint32_t, uint32x2_t, uint32x2_t, vcle, u32)
VCLE_IMPL(int32_t, int32x2_t, uint32x2_t, vcle, s32)
VCLE_IMPL(float32x2_t, float32x2_t, uint32x2_t, vcle, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCLE_IMPL(float16x4_t, float16x4_t, uint16x4_t, vcle, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VCLE_IMPL(uint8_t, uint8x16_t, uint8x16_t, vcleq, u8)
VCLE_IMPL(int8_t, int8x16_t, uint8x16_t, vcleq, s8)
VCLE_IMPL(uint16_t, uint16x8_t, uint16x8_t, vcleq, u16)
VCLE_IMPL(int16_t, int16x8_t, uint16x8_t, vcleq, s16)
VCLE_IMPL(uint32_t, uint32x4_t, uint32x4_t, vcleq, u32)
VCLE_IMPL(int32_t, int32x4_t, uint32x4_t, vcleq, s32)
VCLE_IMPL(float32x4_t, float32x4_t, uint32x4_t, vcleq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCLE_IMPL(float16x8_t, float16x8_t, uint16x8_t, vcleq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCLE_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_CLE_H */
