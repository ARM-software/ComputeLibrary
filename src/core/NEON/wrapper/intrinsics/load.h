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
#ifndef ARM_COMPUTE_WRAPPER_LOAD_H
#define ARM_COMPUTE_WRAPPER_LOAD_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VLOAD_IMPL(stype, vtype, postfix) \
    inline vtype vload(const stype *ptr)  \
    {                                     \
        return vld1_##postfix(ptr);       \
    }

VLOAD_IMPL(uint8_t, uint8x8_t, u8)
VLOAD_IMPL(int8_t, int8x8_t, s8)
VLOAD_IMPL(uint16_t, uint16x4_t, u16)
VLOAD_IMPL(int16_t, int16x4_t, s16)
VLOAD_IMPL(uint32_t, uint32x2_t, u32)
VLOAD_IMPL(int32_t, int32x2_t, s32)
//VLOAD_IMPL(uint64_t, uint64x1_t, u64)
//VLOAD_IMPL(int64_t, int64x1_t, s64)
VLOAD_IMPL(float, float32x2_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VLOAD_IMPL(float16_t, float16x4_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#define VLOADQ_IMPL(stype, vtype, postfix) \
    inline vtype vloadq(const stype *ptr)  \
    {                                      \
        return vld1q_##postfix(ptr);       \
    }

VLOADQ_IMPL(uint8_t, uint8x16_t, u8)
VLOADQ_IMPL(int8_t, int8x16_t, s8)
VLOADQ_IMPL(uint16_t, uint16x8_t, u16)
VLOADQ_IMPL(int16_t, int16x8_t, s16)
VLOADQ_IMPL(uint32_t, uint32x4_t, u32)
VLOADQ_IMPL(int32_t, int32x4_t, s32)
//VLOAD_IMPL(uint64_t, uint64x1_t, u64)
//VLOAD_IMPL(int64_t, int64x1_t, s64)
VLOADQ_IMPL(float, float32x4_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VLOADQ_IMPL(float16_t, float16x8_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#undef VLOAD_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_LOAD_H */
