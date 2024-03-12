/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef SRC_CORE_NEON_WRAPPER_INTRINSICS_SVQADD_H
#define SRC_CORE_NEON_WRAPPER_INTRINSICS_SVQADD_H
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
namespace arm_compute
{
namespace wrapper
{
#define SVQADD_IMPL_F(type, postfix, svppostfix)                        \
    inline type svqadd(const type &val1, const type &val2)              \
    {                                                                   \
        return svadd_##postfix##_z(svptrue_##svppostfix(), val1, val2); \
    }

SVQADD_IMPL_F(svfloat32_t, f32, b32)
SVQADD_IMPL_F(svfloat16_t, f16, b16)
#undef SVQADD_IMPL_F

#define SVQADD_IMPL(type, postfix)                         \
    inline type svqadd(const type &val1, const type &val2) \
    {                                                      \
        return svqadd_##postfix(val1, val2);               \
    }

SVQADD_IMPL(svint32_t, s32)
SVQADD_IMPL(svint16_t, s16)
SVQADD_IMPL(svint8_t, s8)
SVQADD_IMPL(svuint32_t, u32)
SVQADD_IMPL(svuint16_t, u16)
SVQADD_IMPL(svuint8_t, u8)

#undef SVQADD_IMPL
} // namespace wrapper
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* SRC_CORE_NEON_WRAPPER_INTRINSICS_SVQADD_H */