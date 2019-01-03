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
#ifndef __ARM_COMPUTE_WRAPPER_ORR_H__
#define __ARM_COMPUTE_WRAPPER_ORR_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VORR_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vorr(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VORR_IMPL(uint8_t, uint8x8_t, vorr, u8)
VORR_IMPL(int8_t, int8x8_t, vorr, s8)
VORR_IMPL(uint16_t, uint16x4_t, vorr, u16)
VORR_IMPL(int16_t, int16x4_t, vorr, s16)
VORR_IMPL(uint32_t, uint32x2_t, vorr, u32)
VORR_IMPL(int32_t, int32x2_t, vorr, s32)
VORR_IMPL(uint64_t, uint64x1_t, vorr, u64)
VORR_IMPL(int64_t, int64x1_t, vorr, s64)

VORR_IMPL(uint8_t, uint8x16_t, vorrq, u8)
VORR_IMPL(int8_t, int8x16_t, vorrq, s8)
VORR_IMPL(uint16_t, uint16x8_t, vorrq, u16)
VORR_IMPL(int16_t, int16x8_t, vorrq, s16)
VORR_IMPL(uint32_t, uint32x4_t, vorrq, u32)
VORR_IMPL(int32_t, int32x4_t, vorrq, s32)
VORR_IMPL(uint64_t, uint64x2_t, vorrq, u64)
VORR_IMPL(int64_t, int64x2_t, vorrq, s64)

#undef VORR_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_ORR_H__ */
