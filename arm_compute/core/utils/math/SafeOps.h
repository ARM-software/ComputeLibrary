/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_MATH_SAFE_OPS
#define ARM_COMPUTE_UTILS_MATH_SAFE_OPS

#include "arm_compute/core/Error.h"
#include "support/AclRequires.h"

#include <limits>

namespace arm_compute
{
namespace utils
{
namespace math
{
/** Safe integer addition between two integers. In case of an overflow
 *  the numeric max limit is return. In case of an underflow numeric max
 *  limit is return.
 *
 * @tparam T  Integer types to add
 *
 * @param[in] val_a First value to add
 * @param[in] val_b Second value to add
 *
 * @return The addition result
 */
template <typename T, ARM_COMPUTE_REQUIRES_TA(std::is_integral<T>::value)>
T safe_integer_add(T val_a, T val_b)
{
    T result = 0;

    if((val_b > 0) && (val_a > std::numeric_limits<T>::max() - val_b))
    {
        result = std::numeric_limits<T>::max();
    }
    else if((val_b < 0) && (val_a < std::numeric_limits<T>::min() - val_b))
    {
        result = std::numeric_limits<T>::min();
    }
    else
    {
        result = val_a + val_b;
    }

    return result;
}

/** Safe integer subtraction between two integers. In case of an overflow
 *  the numeric max limit is return. In case of an underflow numeric max
 *  limit is return.
 *
 * @tparam T  Integer types to subtract
 *
 * @param[in] val_a Value to subtract from
 * @param[in] val_b Value to subtract
 *
 * @return The subtraction result
 */
template <typename T, ARM_COMPUTE_REQUIRES_TA(std::is_integral<T>::value)>
T safe_integer_sub(T val_a, T val_b)
{
    T result = 0;

    if((val_b < 0) && (val_a > std::numeric_limits<T>::max() + val_b))
    {
        result = std::numeric_limits<T>::max();
    }
    else if((val_b > 0) && (val_a < std::numeric_limits<T>::min() + val_b))
    {
        result = std::numeric_limits<T>::min();
    }
    else
    {
        result = val_a - val_b;
    }

    return result;
}

/** Safe integer multiplication between two integers. In case of an overflow
 *  the numeric max limit is return. In case of an underflow numeric max
 *  limit is return.
 *
 * @tparam T  Integer types to multiply
 *
 * @param[in] val_a First value to multiply
 * @param[in] val_b Second value to multiply
 *
 * @return The multiplication result
 */
template <typename T, ARM_COMPUTE_REQUIRES_TA(std::is_integral<T>::value)>
T safe_integer_mul(T val_a, T val_b)
{
    T result = 0;

    if(val_a > 0)
    {
        if((val_b > 0) && (val_a > (std::numeric_limits<T>::max() / val_b)))
        {
            result = std::numeric_limits<T>::max();
        }
        else if(val_b < (std::numeric_limits<T>::min() / val_a))
        {
            result = std::numeric_limits<T>::min();
        }
        else
        {
            result = val_a * val_b;
        }
    }
    else
    {
        if((val_b > 0) && (val_a < (std::numeric_limits<T>::min() / val_b)))
        {
            result = std::numeric_limits<T>::max();
        }
        else if((val_a != 0) && (val_b < (std::numeric_limits<T>::max() / val_a)))
        {
            result = std::numeric_limits<T>::min();
        }
        else
        {
            result = val_a * val_b;
        }
    }

    return result;
}

/** Safe integer division between two integers. In case of an overflow
 *  the numeric max limit is return. In case of an underflow numeric max
 *  limit is return.
 *
 * @tparam T  Integer types to divide
 *
 * @param[in] val_a Dividend value
 * @param[in] val_b Divisor value
 *
 * @return The quotient
 */
template <typename T, ARM_COMPUTE_REQUIRES_TA(std::is_integral<T>::value)>
T safe_integer_div(T val_a, T val_b)
{
    T result = 0;

    if((val_b == 0) || ((val_a == std::numeric_limits<T>::min()) && (val_b == -1)))
    {
        result = std::numeric_limits<T>::min();
    }
    else
    {
        result = val_a / val_b;
    }

    return result;
}
} // namespace cast
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_MATH_SAFE_OPS */
