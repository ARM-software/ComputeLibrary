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
#ifndef ARM_COMPUTE_WRAPPER_MOVN_H
#define ARM_COMPUTE_WRAPPER_MOVN_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VMOVN_IMPL(dtype, vtype, prefix, postfix) \
    inline dtype vmovn(const vtype &a)            \
    {                                             \
        return prefix##_##postfix(a);             \
    }

VMOVN_IMPL(uint32x2_t, uint64x2_t, vmovn, u64)
VMOVN_IMPL(int32x2_t, int64x2_t, vmovn, s64)
VMOVN_IMPL(uint16x4_t, uint32x4_t, vmovn, u32)
VMOVN_IMPL(int16x4_t, int32x4_t, vmovn, s32)
VMOVN_IMPL(uint8x8_t, uint16x8_t, vmovn, u16)
VMOVN_IMPL(int8x8_t, int16x8_t, vmovn, s16)

#define VQMOVN_IMPL(dtype, vtype, prefix, postfix) \
    inline dtype vqmovn(const vtype &a)            \
    {                                              \
        return prefix##_##postfix(a);              \
    }

VQMOVN_IMPL(uint32x2_t, uint64x2_t, vqmovn, u64)
VQMOVN_IMPL(int32x2_t, int64x2_t, vqmovn, s64)
VQMOVN_IMPL(uint16x4_t, uint32x4_t, vqmovn, u32)
VQMOVN_IMPL(int16x4_t, int32x4_t, vqmovn, s32)
VQMOVN_IMPL(uint8x8_t, uint16x8_t, vqmovn, u16)
VQMOVN_IMPL(int8x8_t, int16x8_t, vqmovn, s16)

#undef VMOVN_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_MOVN_H */
