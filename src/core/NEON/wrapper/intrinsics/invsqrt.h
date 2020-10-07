/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_INVSQRT_H
#define ARM_COMPUTE_WRAPPER_INVSQRT_H

#include "src/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VINVSQRT_IMPL(stype, vtype, prefix, postfix) \
    inline vtype vinvsqrt(const vtype &a)            \
    {                                                \
        return prefix##_##postfix(a);                \
    }

#define VINVSQRT_IMPL_INT(stype, vtype, prefix, postfix) \
    inline vtype vinvsqrt(const vtype &a)                \
    {                                                    \
        ARM_COMPUTE_UNUSED(a);                           \
        ARM_COMPUTE_ERROR("Not supported");              \
    }

VINVSQRT_IMPL(float, float32x2_t, vinvsqrt, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VINVSQRT_IMPL(float16_t, float16x4_t, vinvsqrt, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VINVSQRT_IMPL_INT(int, int32x4_t, vinvsqrt, s32)

VINVSQRT_IMPL(float, float32x4_t, vinvsqrtq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VINVSQRT_IMPL(float16_t, float16x8_t, vinvsqrtq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VINVSQRT_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_INVSQRT_H */
