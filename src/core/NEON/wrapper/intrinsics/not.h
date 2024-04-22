/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_NOT_H
#define ARM_COMPUTE_WRAPPER_NOT_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VNOT_IMPL(stype, vtype, prefix, postfix) \
    inline vtype vnot(const vtype &a)            \
    {                                            \
        return prefix##_##postfix(a);            \
    }

VNOT_IMPL(uint8_t, uint8x8_t, vmvn, u8)
VNOT_IMPL(int8_t, int8x8_t, vmvn, s8)
VNOT_IMPL(uint16_t, uint16x4_t, vmvn, u16)
VNOT_IMPL(int16_t, int16x4_t, vmvn, s16)
VNOT_IMPL(uint32_t, uint32x2_t, vmvn, u32)
VNOT_IMPL(int32_t, int32x2_t, vmvn, s32)
VNOT_IMPL(float32x2_t, float32x2_t, vinv, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VNOT_IMPL(float16x4_t, float16x4_t, vinv, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VNOT_IMPL(uint8_t, uint8x16_t, vmvnq, u8)
VNOT_IMPL(int8_t, int8x16_t, vmvnq, s8)
VNOT_IMPL(uint16_t, uint16x8_t, vmvnq, u16)
VNOT_IMPL(int16_t, int16x8_t, vmvnq, s16)
VNOT_IMPL(uint32_t, uint32x4_t, vmvnq, u32)
VNOT_IMPL(int32_t, int32x4_t, vmvnq, s32)
VNOT_IMPL(float32x4_t, float32x4_t, vinvq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VNOT_IMPL(float16x8_t, float16x8_t, vinvq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VNOT_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_NOT_H */
