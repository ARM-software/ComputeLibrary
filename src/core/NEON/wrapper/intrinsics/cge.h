/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_CGE_H
#define ARM_COMPUTE_WRAPPER_CGE_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VCGE_IMPL(stype, vtype, rtype, prefix, postfix) \
    inline rtype vcge(const vtype &a, const vtype &b)   \
    {                                                   \
        return prefix##_##postfix(a, b);                \
    }

VCGE_IMPL(uint8_t, uint8x8_t, uint8x8_t, vcge, u8)
VCGE_IMPL(int8_t, int8x8_t, uint8x8_t, vcge, s8)
VCGE_IMPL(uint16_t, uint16x4_t, uint16x4_t, vcge, u16)
VCGE_IMPL(int16_t, int16x4_t, uint16x4_t, vcge, s16)
VCGE_IMPL(uint32_t, uint32x2_t, uint32x2_t, vcge, u32)
VCGE_IMPL(int32_t, int32x2_t, uint32x2_t, vcge, s32)
VCGE_IMPL(float32x2_t, float32x2_t, uint32x2_t, vcge, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCGE_IMPL(float16x4_t, float16x4_t, uint16x4_t, vcge, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VCGE_IMPL(uint8_t, uint8x16_t, uint8x16_t, vcgeq, u8)
VCGE_IMPL(int8_t, int8x16_t, uint8x16_t, vcgeq, s8)
VCGE_IMPL(uint16_t, uint16x8_t, uint16x8_t, vcgeq, u16)
VCGE_IMPL(int16_t, int16x8_t, uint16x8_t, vcgeq, s16)
VCGE_IMPL(uint32_t, uint32x4_t, uint32x4_t, vcgeq, u32)
VCGE_IMPL(int32_t, int32x4_t, uint32x4_t, vcgeq, s32)
VCGE_IMPL(float32x4_t, float32x4_t, uint32x4_t, vcgeq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCGE_IMPL(float16x8_t, float16x8_t, uint16x8_t, vcgeq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCGE_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_CGE_H */
