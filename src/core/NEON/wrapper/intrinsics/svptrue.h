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
#ifndef SRC_CORE_NEON_WRAPPER_INTRINSICS_SVPTRUE_H
#define SRC_CORE_NEON_WRAPPER_INTRINSICS_SVPTRUE_H
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
namespace arm_compute
{
namespace wrapper
{
template <size_t element_size>
inline svbool_t  svptrue_size();

template <>
inline svbool_t svptrue_size<64>()
{
    return svptrue_b64();
}

template <>
inline svbool_t svptrue_size<32>()
{
    return svptrue_b32();
}

template <>
inline svbool_t svptrue_size<16>()
{
    return svptrue_b16();
}

template <>
inline svbool_t svptrue_size<8>()
{
    return svptrue_b8();
}

template <typename T>
svbool_t svptrue()
{
    return svptrue_size<sizeof(T) * 8>();
}
} // namespace wrapper
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* SRC_CORE_NEON_WRAPPER_INTRINSICS_SVPTRUE_H */