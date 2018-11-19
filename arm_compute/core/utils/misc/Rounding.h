/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_UTILS_ROUNDING_H__
#define __ARM_COMPUTE_UTILS_ROUNDING_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/Requires.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "support/ToolchainSupport.h"

#include <cmath>

namespace arm_compute
{
namespace utils
{
namespace rounding
{
/** Rounding mode */
enum class RoundingMode
{
    TO_ZERO,             /**< Round towards zero */
    AWAY_FROM_ZERO,      /**< Round away from zero */
    HALF_TO_ZERO,        /**< Round half towards from zero */
    HALF_AWAY_FROM_ZERO, /**< Round half away from zero */
    HALF_UP,             /**< Round half towards positive infinity */
    HALF_DOWN,           /**< Round half towards negative infinity */
    HALF_EVEN            /**< Round half towards nearest even */
};

/** Round floating-point value with round to zero
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round_to_zero(T value)
{
    T res = std::floor(std::fabs(value));
    return (value < 0.f) ? -res : res;
}

/** Round floating-point value with round away from zero
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round_away_from_zero(T value)
{
    T res = std::ceil(std::fabs(value));
    return (value < 0.f) ? -res : res;
}

/** Round floating-point value with half value rounding towards zero.
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round_half_to_zero(T value)
{
    T res = T(std::ceil(std::fabs(value) - 0.5f));
    return (value < 0.f) ? -res : res;
}

/** Round floating-point value with half value rounding away from zero.
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round_half_away_from_zero(T value)
{
    T res = T(std::floor(std::fabs(value) + 0.5f));
    return (value < 0.f) ? -res : res;
}

/** Round floating-point value with half value rounding to positive infinity.
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round_half_up(T value)
{
    return std::floor(value + 0.5f);
}

/** Round floating-point value with half value rounding to negative infinity.
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round_half_down(T value)
{
    return std::ceil(value - 0.5f);
}

/** Round floating-point value with half value rounding to nearest even.
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value   floating-point value to be rounded.
 * @param[in] epsilon precision.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round_half_even(T value, T epsilon = std::numeric_limits<T>::epsilon())
{
    T positive_value = std::abs(value);
    T ipart          = 0;
    std::modf(positive_value, &ipart);
    // If 'value' is exactly halfway between two integers
    if(std::abs(positive_value - (ipart + 0.5f)) < epsilon)
    {
        // If 'ipart' is even then return 'ipart'
        if(std::fmod(ipart, 2.f) < epsilon)
        {
            return support::cpp11::copysign(ipart, value);
        }
        // Else return the nearest even integer
        return support::cpp11::copysign(std::ceil(ipart + 0.5f), value);
    }
    // Otherwise use the usual round to closest
    return support::cpp11::copysign(support::cpp11::round(positive_value), value);
}

/** Round floating-point value given a rounding mode
 *
 * @tparam T Parameter type. Should be of floating point type.
 *
 * @param[in] value         floating-point value to be rounded.
 * @param[in] rounding_mode Rounding mode to use.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, REQUIRES_TA(traits::is_floating_point<T>::value)>
inline T round(T value, RoundingMode rounding_mode)
{
    switch(rounding_mode)
    {
        case RoundingMode::TO_ZERO:
            return round_to_zero(value);
        case RoundingMode::AWAY_FROM_ZERO:
            return round_away_from_zero(value);
        case RoundingMode::HALF_TO_ZERO:
            return round_half_to_zero(value);
        case RoundingMode::HALF_AWAY_FROM_ZERO:
            return round_half_away_from_zero(value);
        case RoundingMode::HALF_UP:
            return round_half_up(value);
        case RoundingMode::HALF_DOWN:
            return round_half_down(value);
        case RoundingMode::HALF_EVEN:
            return round_half_even(value);
        default:
            ARM_COMPUTE_ERROR("Unsupported rounding mode!");
    }
}
} // namespace rounding
} // namespace utils
} // namespace arm_compute
#endif /*__ARM_COMPUTE_UTILS_ROUNDING_H__ */
