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
#ifndef __ARM_COMPUTE_WRAPPER_NEG_H__
#define __ARM_COMPUTE_WRAPPER_NEG_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VNEG_IMPL(vtype, postfix)     \
    inline vtype vneg(const vtype &a) \
    {                                 \
        return vneg_##postfix(a);     \
    }

VNEG_IMPL(int8x8_t, s8)
VNEG_IMPL(int16x4_t, s16)
VNEG_IMPL(int32x2_t, s32)
VNEG_IMPL(float32x2_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VNEG_IMPL(float16x4_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VNEG_IMPL
#define VNEGQ_IMPL(vtype, postfix)     \
    inline vtype vnegq(const vtype &a) \
    {                                  \
        return vnegq_##postfix(a);     \
    }

VNEGQ_IMPL(int8x16_t, s8)
VNEGQ_IMPL(int16x8_t, s16)
VNEGQ_IMPL(int32x4_t, s32)
VNEGQ_IMPL(float32x4_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VNEGQ_IMPL(float16x8_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VNEGQ_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_NEG_H__ */
