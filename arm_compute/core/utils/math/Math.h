/*
 * Copyright (c) 2017-2018, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_MATH_H
#define ARM_COMPUTE_UTILS_MATH_H

namespace arm_compute
{
/** Calculate the rounded up quotient of val / m.
 *
 * @param[in] val Value to divide and round up.
 * @param[in] m   Value to divide by.
 *
 * @return the result.
 */
template <typename S, typename T>
constexpr auto DIV_CEIL(S val, T m) -> decltype((val + m - 1) / m)
{
    return (val + m - 1) / m;
}

/** Computes the smallest number larger or equal to value that is a multiple of divisor.
 *
 * @param[in] value   Lower bound value
 * @param[in] divisor Value to compute multiple of.
 *
 * @return the result.
 */
template <typename S, typename T>
inline auto ceil_to_multiple(S value, T divisor) -> decltype(((value + divisor - 1) / divisor) * divisor)
{
    ARM_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
    return DIV_CEIL(value, divisor) * divisor;
}

/** Computes the largest number smaller or equal to value that is a multiple of divisor.
 *
 * @param[in] value   Upper bound value
 * @param[in] divisor Value to compute multiple of.
 *
 * @return the result.
 */
template <typename S, typename T>
inline auto floor_to_multiple(S value, T divisor) -> decltype((value / divisor) * divisor)
{
    ARM_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
    return (value / divisor) * divisor;
}

}
#endif /*ARM_COMPUTE_UTILS_MATH_H */
