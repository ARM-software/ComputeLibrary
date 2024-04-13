/*
 * Copyright (c) 2018-2020, 2023 Arm Limited.
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
#ifndef ACL_SRC_CORE_NEON_WRAPPER_INTRINSICS_MAX_H
#define ACL_SRC_CORE_NEON_WRAPPER_INTRINSICS_MAX_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VMAX_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vmax(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VMAX_IMPL(uint8_t, uint8x8_t, vmax, u8)
VMAX_IMPL(int8_t, int8x8_t, vmax, s8)
VMAX_IMPL(uint16_t, uint16x4_t, vmax, u16)
VMAX_IMPL(int16_t, int16x4_t, vmax, s16)
VMAX_IMPL(uint32_t, uint32x2_t, vmax, u32)
VMAX_IMPL(int32_t, int32x2_t, vmax, s32)
VMAX_IMPL(float, float32x2_t, vmax, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMAX_IMPL(float16_t, float16x4_t, vmax, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VMAX_IMPL(uint8_t, uint8x16_t, vmaxq, u8)
VMAX_IMPL(int8_t, int8x16_t, vmaxq, s8)
VMAX_IMPL(uint16_t, uint16x8_t, vmaxq, u16)
VMAX_IMPL(int16_t, int16x8_t, vmaxq, s16)
VMAX_IMPL(uint32_t, uint32x4_t, vmaxq, u32)
VMAX_IMPL(int32_t, int32x4_t, vmaxq, s32)
VMAX_IMPL(float, float32x4_t, vmaxq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMAX_IMPL(float16_t, float16x8_t, vmaxq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VMAX_IMPL

#if defined(__aarch64__)
// VMAXV: Across vector max
#define VMAXV_IMPL(stype, vtype, prefix, postfix) \
    inline stype vmaxv(const vtype &a)            \
    {                                             \
        return prefix##_##postfix(a);             \
    }

VMAXV_IMPL(uint8_t, uint8x8_t, vmaxv, u8)
VMAXV_IMPL(int8_t, int8x8_t, vmaxv, s8)
VMAXV_IMPL(uint16_t, uint16x4_t, vmaxv, u16)
VMAXV_IMPL(int16_t, int16x4_t, vmaxv, s16)
VMAXV_IMPL(uint32_t, uint32x2_t, vmaxv, u32)
VMAXV_IMPL(int32_t, int32x2_t, vmaxv, s32)
VMAXV_IMPL(float, float32x2_t, vmaxv, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMAXV_IMPL(float16_t, float16x4_t, vmaxv, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VMAXV_IMPL(uint8_t, uint8x16_t, vmaxvq, u8)
VMAXV_IMPL(int8_t, int8x16_t, vmaxvq, s8)
VMAXV_IMPL(uint16_t, uint16x8_t, vmaxvq, u16)
VMAXV_IMPL(int16_t, int16x8_t, vmaxvq, s16)
VMAXV_IMPL(uint32_t, uint32x4_t, vmaxvq, u32)
VMAXV_IMPL(int32_t, int32x4_t, vmaxvq, s32)
VMAXV_IMPL(float, float32x4_t, vmaxvq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VMAXV_IMPL(float16_t, float16x8_t, vmaxvq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VMAXV_IMPL
#endif // defined(__aarch64__)
} // namespace wrapper
} // namespace arm_compute
#endif // ACL_SRC_CORE_NEON_WRAPPER_INTRINSICS_MAX_H
