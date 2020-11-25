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
#ifndef ARM_COMPUTE_WRAPPER_CGTZ_H
#define ARM_COMPUTE_WRAPPER_CGTZ_H

#ifdef __aarch64__
#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VCGTZ_IMPL(vtype, rtype, prefix, postfix) \
    inline rtype vcgtz(const vtype &a)            \
    {                                             \
        return prefix##_##postfix(a);             \
    }

VCGTZ_IMPL(int8x8_t, uint8x8_t, vcgtz, s8)
VCGTZ_IMPL(int16x4_t, uint16x4_t, vcgtz, s16)
VCGTZ_IMPL(int32x2_t, uint32x2_t, vcgtz, s32)
VCGTZ_IMPL(float32x2_t, uint32x2_t, vcgtz, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCGTZ_IMPL(float16x4_t, uint16x4_t, vcgtz, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VCGTZ_IMPL(int8x16_t, uint8x16_t, vcgtzq, s8)
VCGTZ_IMPL(int16x8_t, uint16x8_t, vcgtzq, s16)
VCGTZ_IMPL(int32x4_t, uint32x4_t, vcgtzq, s32)
VCGTZ_IMPL(float32x4_t, uint32x4_t, vcgtzq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCGTZ_IMPL(float16x8_t, uint16x8_t, vcgtzq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCGTZ_IMPL

} // namespace wrapper
} // namespace arm_compute

#endif // __aarch64__
#endif /* ARM_COMPUTE_WRAPPER_CGTZ_H */
