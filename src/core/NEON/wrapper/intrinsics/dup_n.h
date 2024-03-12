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
#ifndef ARM_COMPUTE_WRAPPER_DUP_N_H
#define ARM_COMPUTE_WRAPPER_DUP_N_H

#include "src/core/NEON/wrapper/traits.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VDUP_N_IMPL(stype, vtype, prefix, postfix, tag) \
    inline vtype vdup_n(stype value, tag)               \
    {                                                   \
        return prefix##_##postfix(value);               \
    }

VDUP_N_IMPL(uint8_t, uint8x8_t, vdup_n, u8, traits::vector_64_tag)
VDUP_N_IMPL(int8_t, int8x8_t, vdup_n, s8, traits::vector_64_tag)
VDUP_N_IMPL(uint16_t, uint16x4_t, vdup_n, u16, traits::vector_64_tag)
VDUP_N_IMPL(int16_t, int16x4_t, vdup_n, s16, traits::vector_64_tag)
VDUP_N_IMPL(uint32_t, uint32x2_t, vdup_n, u32, traits::vector_64_tag)
VDUP_N_IMPL(int32_t, int32x2_t, vdup_n, s32, traits::vector_64_tag)
VDUP_N_IMPL(float, float32x2_t, vdup_n, f32, traits::vector_64_tag)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VDUP_N_IMPL(float16_t, float16x4_t, vdup_n, f16, traits::vector_64_tag)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VDUP_N_IMPL(uint8_t, uint8x16_t, vdupq_n, u8, traits::vector_128_tag)
VDUP_N_IMPL(int8_t, int8x16_t, vdupq_n, s8, traits::vector_128_tag)
VDUP_N_IMPL(uint16_t, uint16x8_t, vdupq_n, u16, traits::vector_128_tag)
VDUP_N_IMPL(int16_t, int16x8_t, vdupq_n, s16, traits::vector_128_tag)
VDUP_N_IMPL(uint32_t, uint32x4_t, vdupq_n, u32, traits::vector_128_tag)
VDUP_N_IMPL(int32_t, int32x4_t, vdupq_n, s32, traits::vector_128_tag)
VDUP_N_IMPL(float, float32x4_t, vdupq_n, f32, traits::vector_128_tag)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VDUP_N_IMPL(float16_t, float16x8_t, vdupq_n, f16, traits::vector_128_tag)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VDUP_N_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_DUP_N_H */
