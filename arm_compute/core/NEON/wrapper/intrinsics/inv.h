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
#ifndef __ARM_COMPUTE_WRAPPER_INV_H__
#define __ARM_COMPUTE_WRAPPER_INV_H__

#include "arm_compute/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VINV_IMPL(vtype, prefix, postfix) \
    inline vtype vinv(const vtype &a)     \
    {                                     \
        return prefix##_##postfix(a);     \
    }

#define VINV_IMPL_INT(vtype, prefix, postfix) \
    inline vtype vinv(const vtype &a)         \
    {                                         \
        ARM_COMPUTE_ERROR("Not supported");   \
    }

VINV_IMPL(float32x2_t, vinv, f32)
VINV_IMPL_INT(int32x2_t, vinv, s32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VINV_IMPL(float16x4_t, vinv, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VINV_IMPL(float32x4_t, vinvq, f32)
VINV_IMPL_INT(int32x4_t, vinvq, s32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VINV_IMPL(float16x8_t, vinvq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VINV_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_INV_H__ */
