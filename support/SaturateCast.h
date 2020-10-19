/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_CAST_SATURATE_CAST_H
#define ARM_COMPUTE_UTILS_CAST_SATURATE_CAST_H

#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "support/Rounding.h"

namespace arm_compute
{
namespace utils
{
namespace cast
{
// *INDENT-OFF*
// clang-format off
// same type
template<typename T,
         typename U,
         typename std::enable_if<std::is_same<T, U>::value, int >::type = 0 >
T saturate_cast(U v)
{
    return v;
}

// signed -> signed widening/same_width
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_signed<U>() &&
                                 std::is_signed<T>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) >= sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(v);
}
// signed -> signed narrowing
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_signed<U>() &&
                                 std::is_signed<T>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) < sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(utility::clamp<U>(v, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
}

// unsigned -> signed widening
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_unsigned<U>() &&
                                 std::is_signed<T>() &&
                                 !std::is_same<T, U>::value &&
                                 (sizeof(T) > sizeof(U)),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(v);
}
// unsigned -> signed narrowing
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_unsigned<U>() &&
                                 std::is_signed<T>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) < sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(std::min<U>(v, std::numeric_limits<T>::max()));
}
// unsigned -> signed same_width
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_unsigned<U>() &&
                                 std::is_signed<T>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) == sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(std::min<U>(v, std::numeric_limits<T>::max()));
}

// signed -> unsigned widening/same width
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_signed<U>() &&
                                 std::is_unsigned<T>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) >= sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(std::max<U>(0, v));
}

// signed -> unsigned narrowing
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_signed<U>() &&
                                 std::is_unsigned<T>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) < sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(utility::clamp<U>(v, 0, std::numeric_limits<T>::max()));
}

// unsigned -> unsigned widening/same width
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_unsigned<T>() &&
                                 std::is_unsigned<U>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) >= sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(v);
}

// unsigned -> unsigned narrowing
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_integral<U>::value &&
                                 std::is_unsigned<T>() &&
                                 std::is_unsigned<U>() &&
                                 !std::is_same<T, U>::value &&
                                 sizeof(T) < sizeof(U),
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(utility::clamp<U>(v, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
}

// float -> int
template<typename T,
         typename U,
         typename std::enable_if<std::is_integral<T>::value &&
                                 traits::is_floating_point<U>::value,
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    int32_t vi = utils::rounding::round_half_away_from_zero(v);
    return saturate_cast<T>(vi);
}

// int -> float
template<typename T,
         typename U,
         typename std::enable_if<traits::is_floating_point<T>::value &&
                                 std::is_integral<U>::value,
                  int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(v);
}

// float -> float
template<typename T,
        typename U,
        typename std::enable_if<traits::is_floating_point<T>::value &&
                                traits::is_floating_point<U>::value,
                int >::type = 0 >
inline T saturate_cast(U v)
{
    return static_cast<T>(v);
}
// clang-format on
// *INDENT-ON*
} // namespace cast
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_CAST_SATURATE_CAST_H */
