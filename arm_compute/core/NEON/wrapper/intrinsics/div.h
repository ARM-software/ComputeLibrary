/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_WRAPPER_DIV_H__
#define __ARM_COMPUTE_WRAPPER_DIV_H__

#include "arm_compute/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#ifdef __aarch64__

#define VDIV_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vdiv(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }
VDIV_IMPL(float32x2_t, float32x2_t, vdiv, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VDIV_IMPL(float16x4_t, float16x4_t, vdiv, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VDIV_IMPL(float32x4_t, float32x4_t, vdivq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VDIV_IMPL(float16x8_t, float16x8_t, vdivq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#else // __aarch64__

#define VDIV_IMPL(stype, vtype, mul_prefix, inv_prefix, postfix)     \
    inline vtype vdiv(const vtype &a, const vtype &b)                \
    {                                                                \
        return mul_prefix##_##postfix(a, inv_prefix##_##postfix(b)); \
    }
VDIV_IMPL(float32x2_t, float32x2_t, vmul, vinv, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VDIV_IMPL(float16x4_t, float16x4_t, vmul, vinv, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VDIV_IMPL(float32x4_t, float32x4_t, vmulq, vinvq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VDIV_IMPL(float16x8_t, float16x8_t, vmulq, vinvq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#endif // __aarch64__

#undef VDIV_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_DIV_H__ */
