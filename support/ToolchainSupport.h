/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_SUPPORT_TOOLCHAINSUPPORT
#define ARM_COMPUTE_SUPPORT_TOOLCHAINSUPPORT

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <arm_neon.h>
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#include "support/Bfloat16.h"
#include "support/Half.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif // M_PI

namespace arm_compute
{
namespace support
{
namespace cpp11
{
#if(__ANDROID__ || BARE_METAL)
template <typename T>
inline T nearbyint(T value)
{
    return static_cast<T>(::nearbyint(value));
}

/** Round floating-point value with half value rounding away from zero.
 *
 * @note This function implements the same behaviour as std::round except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T round(T value)
{
    return ::round(value);
}

/** Round floating-point value with half value rounding away from zero and cast to long
 *
 * @note This function implements the same behaviour as std::lround except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value casted to long
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline long lround(T value)
{
    return ::lround(value);
}

/** Truncate floating-point value.
 *
 * @note This function implements the same behaviour as std::truncate except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] value floating-point value to be truncated.
 *
 * @return Floating-point value of truncated @p value.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T trunc(T value)
{
    return ::trunc(value);
}

/** Composes a floating point value with the magnitude of @p x and the sign of @p y.
 *
 * @note This function implements the same behaviour as std::copysign except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] x value that contains the magnitude to be used in constructing the result.
 * @param[in] y value that contains the sign to be used in construct in the result.
 *
 * @return Floating-point value with magnitude of @p x and sign of @p y.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T copysign(T x, T y)
{
    return ::copysign(x, y);
}

/** Computes (x*y) + z as if to infinite precision and rounded only once to fit the result type.
 *
 * @note This function implements the same behaviour as std::fma except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] x floating-point value
 * @param[in] y floating-point value
 * @param[in] z floating-point value
 *
 * @return Result floating point value equal to (x*y) + z.c
 */
template < typename T, typename = typename std::enable_if < std::is_floating_point<T>::value
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                            || std::is_same<T, float16_t>::value
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                            >::type >
inline T fma(T x, T y, T z)
{
    return ::fma(x, y, z);
}

/** Loads the data from the given location, converts them to character string equivalents
 *  and writes the result to a character string buffer.
 *
 * @param[in] s    Pointer to a character string to write to
 * @param[in] n    Up to buf_size - 1 characters may be written, plus the null ending character
 * @param[in] fmt  Pointer to a null-ended multibyte string specifying how to interpret the data.
 * @param[in] args Arguments forwarded to snprintf.
 *
 * @return  Number of characters that would have been written for a sufficiently large buffer
 *          if successful (not including the ending null character), or a negative value if an error occurred.
 */
template <typename... Ts>
inline int snprintf(char *s, size_t n, const char *fmt, Ts &&... args)
{
    return ::snprintf(s, n, fmt, std::forward<Ts>(args)...);
}
#else /* (__ANDROID__ || BARE_METAL) */
/** Rounds the floating-point argument arg to an integer value in floating-point format, using the current rounding mode.
 *
 * @note This function acts as a convenience wrapper around std::nearbyint. The
 *       latter is missing in some Android toolchains.
 *
 * @param[in] value Value to be rounded.
 *
 * @return The rounded value.
 */
template <typename T>
inline T nearbyint(T value)
{
    return static_cast<T>(std::nearbyint(value));
}

/** Round floating-point value with half value rounding away from zero.
 *
 * @note This function implements the same behaviour as std::round except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T round(T value)
{
    //Workaround Valgrind's mismatches: when running from Valgrind the call to std::round(-4.500000) == -4.000000 instead of 5.00000
    return (value < 0.f) ? static_cast<int>(value - 0.5f) : static_cast<int>(value + 0.5f);
}

/** Round floating-point value with half value rounding away from zero and cast to long
 *
 * @note This function implements the same behaviour as std::lround except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value casted to long
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline long lround(T value)
{
    return std::lround(value);
}

/** Truncate floating-point value.
 *
 * @note This function implements the same behaviour as std::truncate except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] value floating-point value to be truncated.
 *
 * @return Floating-point value of truncated @p value.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T trunc(T value)
{
    return std::trunc(value);
}

/** Composes a floating point value with the magnitude of @p x and the sign of @p y.
 *
 * @note This function implements the same behaviour as std::copysign except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] x value that contains the magnitude to be used in constructing the result.
 * @param[in] y value that contains the sign to be used in construct in the result.
 *
 * @return Floating-point value with magnitude of @p x and sign of @p y.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T copysign(T x, T y)
{
    return std::copysign(x, y);
}

/** Computes (x*y) + z as if to infinite precision and rounded only once to fit the result type.
 *
 * @note This function implements the same behaviour as std::fma except that it doesn't
 *       support Integral type. The latter is not in the namespace std in some Android toolchains.
 *
 * @param[in] x floating-point value
 * @param[in] y floating-point value
 * @param[in] z floating-point value
 *
 * @return Result floating point value equal to (x*y) + z.
 */
template < typename T, typename = typename std::enable_if < std::is_floating_point<T>::value
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                            || std::is_same<T, float16_t>::value
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                            >::type >
inline T fma(T x, T y, T z)
{
    return std::fma(x, y, z);
}

/** Loads the data from the given location, converts them to character string equivalents
 *  and writes the result to a character string buffer.
 *
 * @param[in] s    Pointer to a character string to write to
 * @param[in] n    Up to buf_size - 1 characters may be written, plus the null ending character
 * @param[in] fmt  Pointer to a null-ended multibyte string specifying how to interpret the data.
 * @param[in] args Arguments forwarded to std::snprintf.
 *
 * @return  Number of characters that would have been written for a sufficiently large buffer
 *          if successful (not including the ending null character), or a negative value if an error occurred.
 */
template <typename... Ts>
inline int snprintf(char *s, std::size_t n, const char *fmt, Ts &&... args)
{
    return std::snprintf(s, n, fmt, std::forward<Ts>(args)...);
}
#endif /* (__ANDROID__ || BARE_METAL) */

// std::numeric_limits<T>::lowest
template <typename T>
inline T lowest()
{
    return std::numeric_limits<T>::lowest();
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline __fp16 lowest<__fp16>()
{
    return std::numeric_limits<half_float::half>::lowest();
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <>
inline bfloat16 lowest<bfloat16>()
{
    return bfloat16::lowest();
}

// std::isfinite
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline bool isfinite(T value)
{
    return std::isfinite(static_cast<double>(value));
}

inline bool isfinite(half_float::half value)
{
    return half_float::isfinite(value);
}

inline bool isfinite(bfloat16 value)
{
    return std::isfinite(float(value));
}

// std::signbit
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline bool signbit(T value)
{
    return std::signbit(static_cast<double>(value));
}

inline bool signbit(half_float::half value)
{
    return half_float::signbit(value);
}

inline bool signbit(bfloat16 value)
{
    return std::signbit(float(value));
}
} // namespace cpp11
} // namespace support
} // namespace arm_compute
#endif /* ARM_COMPUTE_SUPPORT_TOOLCHAINSUPPORT */
