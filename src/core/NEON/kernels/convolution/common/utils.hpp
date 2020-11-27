/*
 * Copyright (c) 2017-2018 Arm Limited.
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

#pragma once

#include <limits>

void PrintMatrix(const float *const m, const int M, const int N, const int row_stride);

constexpr inline int iceildiv(const int a, const int b)
{
    return (a + b - 1) / b;
}

template <typename T>
inline T roundup(const T a, const T b)
{
    return b * iceildiv(a, b);
}

template<typename T>
struct TypeBounds
{
    static constexpr T lower() noexcept { return std::numeric_limits<T>::has_infinity
                                                 ? -std::numeric_limits<T>::infinity()
                                                 : std::numeric_limits<T>::lowest(); };
    static constexpr T upper() noexcept { return std::numeric_limits<T>::has_infinity
                                                 ? std::numeric_limits<T>::infinity()
                                                 : std::numeric_limits<T>::max(); };
};

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template<>
struct TypeBounds<__fp16>
{
    static constexpr __fp16 lower() noexcept { return -std::numeric_limits<float>::infinity(); };
    static constexpr __fp16 upper() noexcept { return std::numeric_limits<float>::infinity(); }
};
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
