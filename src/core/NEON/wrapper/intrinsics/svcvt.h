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
#ifndef SRC_CORE_NEON_WRAPPER_INTRINSICS_SVCVT_H
#define SRC_CORE_NEON_WRAPPER_INTRINSICS_SVCVT_H
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
namespace arm_compute
{
namespace wrapper
{
#define SVCVT_Z_TO_F32_IMPL(vtype)                                                                                        \
    template <typename T>                                                                                                 \
    inline typename std::enable_if<std::is_same<T, float>::value, svfloat32_t>::type svcvt_z(svbool_t pg, const vtype &a) \
    {                                                                                                                     \
        return svcvt_f32_z(pg, a);                                                                                        \
    }

SVCVT_Z_TO_F32_IMPL(svuint32_t)
SVCVT_Z_TO_F32_IMPL(svint32_t)
SVCVT_Z_TO_F32_IMPL(svfloat16_t)

#undef SVCVT_Z_TO_F32_IMPL

#define SVCVT_Z_TO_F16_IMPL(vtype)                                                                                            \
    template <typename T>                                                                                                     \
    inline typename std::enable_if<std::is_same<T, float16_t>::value, svfloat16_t>::type svcvt_z(svbool_t pg, const vtype &a) \
    {                                                                                                                         \
        return svcvt_f16_z(pg, a);                                                                                            \
    }

SVCVT_Z_TO_F16_IMPL(svuint32_t)
SVCVT_Z_TO_F16_IMPL(svint32_t)
SVCVT_Z_TO_F16_IMPL(svfloat32_t)

#undef SVCVT_Z_TO_F16_IMPL

#define SVCVT_Z_TO_S32_IMPL(vtype)                                                                                        \
    template <typename T>                                                                                                 \
    inline typename std::enable_if<std::is_same<T, int32_t>::value, svint32_t>::type svcvt_z(svbool_t pg, const vtype &a) \
    {                                                                                                                     \
        return svcvt_s32_z(pg, a);                                                                                        \
    }

SVCVT_Z_TO_S32_IMPL(svfloat16_t)
SVCVT_Z_TO_S32_IMPL(svfloat32_t)

#undef SVCVT_Z_TO_S32_IMPL

} // namespace wrapper
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* SRC_CORE_NEON_WRAPPER_INTRINSICS_SVCVT_H */