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
#ifndef __ARM_COMPUTE_WRAPPER_PADD_H__
#define __ARM_COMPUTE_WRAPPER_PADD_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
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
#endif /* __ARM_COMPUTE_WRAPPER_PADD_H__ */
