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
#ifndef __ARM_COMPUTE_WRAPPER_NEG_H__
#define __ARM_COMPUTE_WRAPPER_NEG_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VNEG_IMPL(vtype, prefix, postfix) \
    inline vtype vneg(const vtype &a)     \
    {                                     \
        return prefix##_##postfix(a);     \
    }

VNEG_IMPL(int8x8_t, vneg, s8)
VNEG_IMPL(int16x4_t, vneg, s16)
VNEG_IMPL(int32x2_t, vneg, s32)
VNEG_IMPL(float32x2_t, vneg, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VNEG_IMPL(float16x4_t, vneg, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VNEG_IMPL(int8x16_t, vnegq, s8)
VNEG_IMPL(int16x8_t, vnegq, s16)
VNEG_IMPL(int32x4_t, vnegq, s32)
VNEG_IMPL(float32x4_t, vnegq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VNEG_IMPL(float16x8_t, vnegq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VNEG_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_NEG_H__ */
