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
#ifndef ARM_COMPUTE_WRAPPER_AND_H
#define ARM_COMPUTE_WRAPPER_AND_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VAND_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vand(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VAND_IMPL(uint8_t, uint8x8_t, vand, u8)
VAND_IMPL(int8_t, int8x8_t, vand, s8)
VAND_IMPL(uint16_t, uint16x4_t, vand, u16)
VAND_IMPL(int16_t, int16x4_t, vand, s16)
VAND_IMPL(uint32_t, uint32x2_t, vand, u32)
VAND_IMPL(int32_t, int32x2_t, vand, s32)
VAND_IMPL(uint64_t, uint64x1_t, vand, u64)
VAND_IMPL(int64_t, int64x1_t, vand, s64)

VAND_IMPL(uint8_t, uint8x16_t, vandq, u8)
VAND_IMPL(int8_t, int8x16_t, vandq, s8)
VAND_IMPL(uint16_t, uint16x8_t, vandq, u16)
VAND_IMPL(int16_t, int16x8_t, vandq, s16)
VAND_IMPL(uint32_t, uint32x4_t, vandq, u32)
VAND_IMPL(int32_t, int32x4_t, vandq, s32)
VAND_IMPL(uint64_t, uint64x2_t, vandq, u64)
VAND_IMPL(int64_t, int64x2_t, vandq, s64)

#undef VAND_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_AND_H */
