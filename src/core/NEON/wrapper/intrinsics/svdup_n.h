/*
 * Copyright (c) 2020, 2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_WRAPPER_INTRINSICS_SVDUP_N_H
#define SRC_CORE_NEON_WRAPPER_INTRINSICS_SVDUP_N_H
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
namespace arm_compute
{
namespace wrapper
{
#define SVDUP_N_IMPL(etype, vtype, postfix) \
    inline vtype svdup_n(etype a)           \
    {                                       \
        return svdup_n_##postfix(a);        \
    }

SVDUP_N_IMPL(int8_t, svint8_t, s8)
SVDUP_N_IMPL(int16_t, svint16_t, s16)
SVDUP_N_IMPL(int32_t, svint32_t, s32)
SVDUP_N_IMPL(int64_t, svint64_t, s64)
SVDUP_N_IMPL(uint8_t, svuint8_t, u8)
SVDUP_N_IMPL(uint16_t, svuint16_t, u16)
SVDUP_N_IMPL(uint32_t, svuint32_t, u32)
SVDUP_N_IMPL(uint64_t, svuint64_t, u64)
SVDUP_N_IMPL(float16_t, svfloat16_t, f16)
SVDUP_N_IMPL(float, svfloat32_t, f32)
SVDUP_N_IMPL(float64_t, svfloat64_t, f64)
#if __ARM_FEATURE_SVE_BF16
SVDUP_N_IMPL(bfloat16_t, svbfloat16_t, bf16)
#endif // #if __ARM_FEATURE_SVE_BF16

#undef SVDUP_N_IMPL

} // namespace wrapper
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* SRC_CORE_NEON_WRAPPER_INTRINSICS_SVDUP_N_H */
