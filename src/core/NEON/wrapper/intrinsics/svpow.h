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
#ifndef SRC_CORE_NEON_WRAPPER_INTRINSICS_SVPOW_H
#define SRC_CORE_NEON_WRAPPER_INTRINSICS_SVPOW_H
#if defined(__ARM_FEATURE_SVE)
#include "src/core/NEON/SVEMath.h"
namespace arm_compute
{
namespace wrapper
{
#define SVPOW_Z_IMPL(type, postfix)                                \
    inline type svpow_z(svbool_t pg, const type &a, const type &b) \
    {                                                              \
        return svpow_##postfix##_z(pg, a, b);                      \
    }

#define SVPOW_Z_IMPL_INT(type, postfix)                            \
    inline type svpow_z(svbool_t pg, const type &a, const type &b) \
    {                                                              \
        ARM_COMPUTE_UNUSED(pg, a, b);                              \
        ARM_COMPUTE_ERROR("Not supported");                        \
    }

SVPOW_Z_IMPL(svfloat32_t, f32)
SVPOW_Z_IMPL(svfloat16_t, f16)
SVPOW_Z_IMPL_INT(svint16_t, s16)

#undef SVPOW_Z_IMPL

} // namespace wrapper
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* SRC_CORE_NEON_WRAPPER_INTRINSICS_SVPOW_H */