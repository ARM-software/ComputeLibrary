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
#ifndef __ARM_COMPUTE_WRAPPER_CEQ_H__
#define __ARM_COMPUTE_WRAPPER_CEQ_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VCEQ_IMPL(votype, vtype, prefix, postfix)      \
    inline votype vceq(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

VCEQ_IMPL(uint8x8_t, uint8x8_t, vceq, u8)
VCEQ_IMPL(uint8x8_t, int8x8_t, vceq, s8)
VCEQ_IMPL(uint16x4_t, uint16x4_t, vceq, u16)
VCEQ_IMPL(uint16x4_t, int16x4_t, vceq, s16)
VCEQ_IMPL(uint32x2_t, uint32x2_t, vceq, u32)
VCEQ_IMPL(uint32x2_t, int32x2_t, vceq, s32)
VCEQ_IMPL(uint32x2_t, float32x2_t, vceq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCEQ_IMPL(uint16x4_t, float16x4_t, vceq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VCEQ_IMPL(uint8x16_t, uint8x16_t, vceqq, u8)
VCEQ_IMPL(uint8x16_t, int8x16_t, vceqq, s8)
VCEQ_IMPL(uint16x8_t, uint16x8_t, vceqq, u16)
VCEQ_IMPL(uint16x8_t, int16x8_t, vceqq, s16)
VCEQ_IMPL(uint32x4_t, uint32x4_t, vceqq, u32)
VCEQ_IMPL(uint32x4_t, int32x4_t, vceqq, s32)
VCEQ_IMPL(uint32x4_t, float32x4_t, vceqq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCEQ_IMPL(uint16x8_t, float16x8_t, vceqq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCEQ_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_CEQ_H__ */
