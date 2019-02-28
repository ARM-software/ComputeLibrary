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
#ifndef __ARM_COMPUTE_WRAPPER_CGT_H__
#define __ARM_COMPUTE_WRAPPER_CGT_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VCGT_IMPL(rtype, vtype, prefix, postfix)      \
    inline rtype vcgt(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VCGT_IMPL(uint8x8_t, uint8x8_t, vcgt, u8)
VCGT_IMPL(uint8x8_t, int8x8_t, vcgt, s8)
VCGT_IMPL(uint16x4_t, uint16x4_t, vcgt, u16)
VCGT_IMPL(uint16x4_t, int16x4_t, vcgt, s16)
VCGT_IMPL(uint32x2_t, uint32x2_t, vcgt, u32)
VCGT_IMPL(uint32x2_t, int32x2_t, vcgt, s32)
VCGT_IMPL(uint32x2_t, float32x2_t, vcgt, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCGT_IMPL(uint16x4_t, float16x4_t, vcgt, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VCGT_IMPL(uint8x16_t, uint8x16_t, vcgtq, u8)
VCGT_IMPL(uint8x16_t, int8x16_t, vcgtq, s8)
VCGT_IMPL(uint16x8_t, uint16x8_t, vcgtq, u16)
VCGT_IMPL(uint16x8_t, int16x8_t, vcgtq, s16)
VCGT_IMPL(uint32x4_t, uint32x4_t, vcgtq, u32)
VCGT_IMPL(uint32x4_t, int32x4_t, vcgtq, s32)
VCGT_IMPL(uint32x4_t, float32x4_t, vcgtq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCGT_IMPL(uint16x8_t, float16x8_t, vcgtq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCGT_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_CGT_H__ */
