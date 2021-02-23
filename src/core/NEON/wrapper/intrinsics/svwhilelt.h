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
#ifndef SRC_CORE_NEON_WRAPPER_INTRINSICS_SVWHILELT_H
#define SRC_CORE_NEON_WRAPPER_INTRINSICS_SVWHILELT_H
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
namespace arm_compute
{
namespace wrapper
{
#define SVWHILELT_IMPL(type)                           \
    template <size_t element_size>                     \
    inline svbool_t svwhilelt_size(type a, type b);    \
    \
    template <>                                        \
    inline svbool_t svwhilelt_size<64>(type a, type b) \
    {                                                  \
        return svwhilelt_b64(a, b);                    \
    }                                                  \
    template <>                                        \
    inline svbool_t svwhilelt_size<32>(type a, type b) \
    {                                                  \
        return svwhilelt_b32(a, b);                    \
    }                                                  \
    template <>                                        \
    inline svbool_t svwhilelt_size<16>(type a, type b) \
    {                                                  \
        return svwhilelt_b16(a, b);                    \
    }                                                  \
    template <>                                        \
    inline svbool_t svwhilelt_size<8>(type a, type b)  \
    {                                                  \
        return svwhilelt_b8(a, b);                     \
    }

SVWHILELT_IMPL(int32_t)
SVWHILELT_IMPL(uint32_t)
SVWHILELT_IMPL(int64_t)
SVWHILELT_IMPL(uint64_t)

#undef SVWHILELT_IMPL

template <typename ScalarType, typename IndexType>
inline svbool_t svwhilelt(IndexType a, IndexType b)
{
    return svwhilelt_size<sizeof(ScalarType) * 8>(a, b);
}
} // namespace wrapper
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* SRC_CORE_NEON_WRAPPER_INTRINSICS_SVWHILELT_H */