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
#ifndef __ARM_COMPUTE_WRAPPER_CLT_H__
#define __ARM_COMPUTE_WRAPPER_CLT_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VCLT_IMPL(votype, vtype, prefix, postfix)      \
    inline votype vclt(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

VCLT_IMPL(uint8x8_t, uint8x8_t, vclt, u8)
VCLT_IMPL(uint8x8_t, int8x8_t, vclt, s8)
VCLT_IMPL(uint16x4_t, uint16x4_t, vclt, u16)
VCLT_IMPL(uint16x4_t, int16x4_t, vclt, s16)
VCLT_IMPL(uint32x2_t, uint32x2_t, vclt, u32)
VCLT_IMPL(uint32x2_t, int32x2_t, vclt, s32)
VCLT_IMPL(uint32x2_t, float32x2_t, vclt, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCLT_IMPL(uint16x4_t, float16x4_t, vclt, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VCLT_IMPL(uint8x16_t, uint8x16_t, vcltq, u8)
VCLT_IMPL(uint8x16_t, int8x16_t, vcltq, s8)
VCLT_IMPL(uint16x8_t, uint16x8_t, vcltq, u16)
VCLT_IMPL(uint16x8_t, int16x8_t, vcltq, s16)
VCLT_IMPL(uint32x4_t, uint32x4_t, vcltq, u32)
VCLT_IMPL(uint32x4_t, int32x4_t, vcltq, s32)
VCLT_IMPL(uint32x4_t, float32x4_t, vcltq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCLT_IMPL(uint16x8_t, float16x8_t, vcltq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCLT_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_CLT_H__ */
