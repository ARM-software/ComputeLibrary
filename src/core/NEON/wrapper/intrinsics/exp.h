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
#ifndef ARM_COMPUTE_WRAPPER_EXP_H
#define ARM_COMPUTE_WRAPPER_EXP_H

#include "src/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VEXPQ_IMPL(vtype, postfix)     \
    inline vtype vexpq(const vtype &a) \
    {                                  \
        return vexpq_##postfix(a);     \
    }

#define VEXPQ_IMPL_INT(vtype, postfix)      \
    inline vtype vexpq(const vtype &a)      \
    {                                       \
        ARM_COMPUTE_UNUSED(a);              \
        ARM_COMPUTE_ERROR("Not supported"); \
    }

VEXPQ_IMPL(float32x4_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VEXPQ_IMPL(float16x8_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VEXPQ_IMPL_INT(int32x4_t, s32)
#undef VEXPQ_IMPL

} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_EXP_H */
