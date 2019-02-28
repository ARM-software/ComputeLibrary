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
#ifndef __ARM_COMPUTE_WRAPPER_REV64_H__
#define __ARM_COMPUTE_WRAPPER_REV64_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VREV64_IMPL(vtype, prefix, postfix) \
    inline vtype vrev64(const vtype &a)     \
    {                                       \
        return prefix##_##postfix(a);       \
    }

VREV64_IMPL(uint8x8_t, vrev64, u8)
VREV64_IMPL(int8x8_t, vrev64, s8)
VREV64_IMPL(uint16x4_t, vrev64, u16)
VREV64_IMPL(int16x4_t, vrev64, s16)
VREV64_IMPL(uint32x2_t, vrev64, u32)
VREV64_IMPL(int32x2_t, vrev64, s32)
VREV64_IMPL(float32x2_t, vrev64, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VREV64_IMPL(float16x4_t, vrev64, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VREV64_IMPL(uint8x16_t, vrev64q, u8)
VREV64_IMPL(int8x16_t, vrev64q, s8)
VREV64_IMPL(uint16x8_t, vrev64q, u16)
VREV64_IMPL(int16x8_t, vrev64q, s16)
VREV64_IMPL(uint32x4_t, vrev64q, u32)
VREV64_IMPL(int32x4_t, vrev64q, s32)
VREV64_IMPL(float32x4_t, vrev64q, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VREV64_IMPL(float16x8_t, vrev64q, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VREV64_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_REV64_H__ */
