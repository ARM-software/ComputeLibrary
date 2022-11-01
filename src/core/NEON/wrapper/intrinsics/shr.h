/*
 * Copyright (c) 2022 Arm Limited.
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

#ifndef ARM_COMPUTE_WRAPPER_SHR_H
#define ARM_COMPUTE_WRAPPER_SHR_H

#include <arm_neon.h>
#include <type_traits>

namespace arm_compute
{
namespace wrapper
{
#define VQRSHRN_IMPL(half_vtype, vtype, prefix, postfix) \
    template <int b>                                     \
    inline half_vtype vqrshrn(const vtype &a)            \
    {                                                    \
        return prefix##_##postfix(a, b);                 \
    }
VQRSHRN_IMPL(int8x8_t, int16x8_t, vqrshrn_n, s16)
VQRSHRN_IMPL(uint8x8_t, uint16x8_t, vqrshrn_n, u16)
VQRSHRN_IMPL(int16x4_t, int32x4_t, vqrshrn_n, s32)
VQRSHRN_IMPL(uint16x4_t, uint32x4_t, vqrshrn_n, u32)
VQRSHRN_IMPL(int32x2_t, int64x2_t, vqrshrn_n, s64)
VQRSHRN_IMPL(uint32x2_t, uint64x2_t, vqrshrn_n, u64)

#undef VQRSHRN_IMPL

#ifdef __aarch64__
#define VQRSHRN_SCALAR_IMPL(half_vtype, vtype, prefix, postfix) \
    template <int b>                                            \
    inline half_vtype vqrshrn(const vtype &a)                   \
    {                                                           \
        return prefix##_##postfix(a, b);                        \
    }

VQRSHRN_SCALAR_IMPL(int8_t, int16_t, vqrshrnh_n, s16)
VQRSHRN_SCALAR_IMPL(uint8_t, uint16_t, vqrshrnh_n, u16)
VQRSHRN_SCALAR_IMPL(int16_t, int32_t, vqrshrns_n, s32)
VQRSHRN_SCALAR_IMPL(uint16_t, uint32_t, vqrshrns_n, u32)
VQRSHRN_SCALAR_IMPL(int32_t, int64_t, vqrshrnd_n, s64)
VQRSHRN_SCALAR_IMPL(uint32_t, uint64_t, vqrshrnd_n, u64)

#undef VQRSHRN_SCALAR_IMPL
#endif // __aarch64__

// This function is the mixed version of VQRSHRN and VQRSHRUN.
// The input vector is always signed integer, while the returned vector
// can be either signed or unsigned depending on the signedness of scalar type T.
#define VQRSHRN_EX_IMPL(half_vtype, vtype, prefix_signed, prefix_unsigned, postfix)                              \
    template <int b, typename T>                                                                                 \
    inline typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, half_vtype>::type     \
    vqrshrn_ex(const vtype &a)                                                                                   \
    {                                                                                                            \
        return prefix_signed##_##postfix(a, b);                                                                  \
    }                                                                                                            \
    \
    template <int b, typename T>                                                                                 \
    inline typename std::enable_if<std::is_integral<T>::value && !std::is_signed<T>::value, u##half_vtype>::type \
    vqrshrn_ex(const vtype &a)                                                                                   \
    {                                                                                                            \
        return prefix_unsigned##_##postfix(a, b);                                                                \
    }
VQRSHRN_EX_IMPL(int8x8_t, int16x8_t, vqrshrn_n, vqrshrun_n, s16)
VQRSHRN_EX_IMPL(int16x4_t, int32x4_t, vqrshrn_n, vqrshrun_n, s32)
VQRSHRN_EX_IMPL(int32x2_t, int64x2_t, vqrshrn_n, vqrshrun_n, s64)
#undef VQRSHRN_EX_IMPL

#define VSHR_IMPL(vtype, prefix, postfix) \
    template <int b>                      \
    inline vtype vshr_n(const vtype &a)   \
    {                                     \
        return prefix##_##postfix(a, b);  \
    }
VSHR_IMPL(uint8x8_t, vshr_n, u8)
VSHR_IMPL(int8x8_t, vshr_n, s8)
#undef VSHR_IMPL

#define VSHRQ_IMPL(vtype, prefix, postfix) \
    template <int b>                       \
    inline vtype vshrq_n(const vtype &a)   \
    {                                      \
        return prefix##_##postfix(a, b);   \
    }
VSHRQ_IMPL(uint32x4_t, vshrq_n, u32)
VSHRQ_IMPL(int32x4_t, vshrq_n, s32)
#undef VSHRQ_IMPL

#ifdef __aarch64__
#define VSHRQ_SCALAR_IMPL(vtype, prefix, postfix) \
    template <int b>                              \
    inline vtype vshrq_n(const vtype &a)          \
    {                                             \
        return prefix##_##postfix(a, b);          \
    }
VSHRQ_SCALAR_IMPL(uint32_t, vshrd_n, u64)
VSHRQ_SCALAR_IMPL(int32_t, vshrd_n, s64)

#undef VSHRQ_SCALAR_IMPL
#endif // __aarch64__

#ifdef __aarch64__
#define VQRSHRN_EX_SCALAR_IMPL(half_vtype, vtype, prefix_signed, prefix_unsigned, postfix)                       \
    template <int b, typename T>                                                                                 \
    inline typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, half_vtype>::type     \
    vqrshrn_ex(const vtype &a)                                                                                   \
    {                                                                                                            \
        return prefix_signed##_##postfix(a, b);                                                                  \
    }                                                                                                            \
    \
    template <int b, typename T>                                                                                 \
    inline typename std::enable_if<std::is_integral<T>::value && !std::is_signed<T>::value, u##half_vtype>::type \
    vqrshrn_ex(const vtype &a)                                                                                   \
    {                                                                                                            \
        return prefix_unsigned##_##postfix(a, b);                                                                \
    }

VQRSHRN_EX_SCALAR_IMPL(int8_t, int16_t, vqrshrnh_n, vqrshrunh_n, s16)
VQRSHRN_EX_SCALAR_IMPL(int16_t, int32_t, vqrshrns_n, vqrshruns_n, s32)
VQRSHRN_EX_SCALAR_IMPL(int32_t, int64_t, vqrshrnd_n, vqrshrund_n, s64)

#undef VQRSHRN_EX_IMPL
#endif // __aarch64__

} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_SHR_H */
