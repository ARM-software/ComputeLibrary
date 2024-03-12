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
#ifndef ARM_COMPUTE_WRAPPER_GET_HIGH_H
#define ARM_COMPUTE_WRAPPER_GET_HIGH_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VGETHIGH_IMPL(half_vtype, vtype, postfix) \
    inline half_vtype vgethigh(const vtype val)   \
    {                                             \
        return vget_high_##postfix(val);          \
    }

VGETHIGH_IMPL(uint8x8_t, uint8x16_t, u8)
VGETHIGH_IMPL(int8x8_t, int8x16_t, s8)
VGETHIGH_IMPL(uint16x4_t, uint16x8_t, u16)
VGETHIGH_IMPL(int16x4_t, int16x8_t, s16)
VGETHIGH_IMPL(uint32x2_t, uint32x4_t, u32)
VGETHIGH_IMPL(int32x2_t, int32x4_t, s32)
VGETHIGH_IMPL(float32x2_t, float32x4_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VGETHIGH_IMPL(float16x4_t, float16x8_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VGETHIGH_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_GET_HIGH_H */
