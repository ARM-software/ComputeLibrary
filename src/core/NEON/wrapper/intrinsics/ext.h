/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_EXT_H
#define ARM_COMPUTE_WRAPPER_EXT_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VEXT_IMPL(vtype, prefix, postfix, size)            \
    inline vtype vext_##size(vtype value_a, vtype value_b) \
    {                                                      \
        return prefix##_##postfix(value_a, value_b, size); \
    }

VEXT_IMPL(uint8x8_t, vext, u8, 1)
VEXT_IMPL(uint8x8_t, vext, u8, 2)
VEXT_IMPL(int8x8_t, vext, s8, 1)
VEXT_IMPL(int8x8_t, vext, s8, 2)
VEXT_IMPL(uint16x4_t, vext, u16, 1)
VEXT_IMPL(uint16x4_t, vext, u16, 2)
VEXT_IMPL(int16x4_t, vext, s16, 1)
VEXT_IMPL(int16x4_t, vext, s16, 2)

VEXT_IMPL(uint8x16_t, vextq, u8, 1)
VEXT_IMPL(uint8x16_t, vextq, u8, 2)
VEXT_IMPL(int8x16_t, vextq, s8, 1)
VEXT_IMPL(int8x16_t, vextq, s8, 2)
VEXT_IMPL(uint16x8_t, vextq, u16, 1)
VEXT_IMPL(uint16x8_t, vextq, u16, 2)
VEXT_IMPL(int16x8_t, vextq, s16, 1)
VEXT_IMPL(int16x8_t, vextq, s16, 2)
VEXT_IMPL(int32x4_t, vextq, s32, 1)
VEXT_IMPL(int32x4_t, vextq, s32, 2)

#undef VEXT_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_EXT_H */
