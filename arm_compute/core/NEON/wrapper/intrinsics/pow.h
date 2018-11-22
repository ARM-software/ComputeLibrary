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
#ifndef __ARM_COMPUTE_WRAPPER_POW_H__
#define __ARM_COMPUTE_WRAPPER_POW_H__

#include "arm_compute/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VPOW_IMPL(vtype, prefix, postfix)             \
    inline vtype vpow(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VPOW_IMPL(float32x4_t, vpowq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VPOW_IMPL(float16x8_t, vpowq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VPOW_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_POW_H__ */
