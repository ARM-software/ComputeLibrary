/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_TOOLCHAINSUPPORT
#define ARM_COMPUTE_TEST_TOOLCHAINSUPPORT

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>

#include "support/Half.h"

namespace arm_compute
{
namespace support
{
namespace cpp11
{
enum class NumericBase
{
    BASE_10,
    BASE_16
};

/** Convert string values to integer.
 *
 * @note This function implements the same behaviour as std::stoi. The latter
 *       is missing in some Android toolchains.
 *
 * @param[in] str  String to be converted to int.
 * @param[in] pos  If idx is not a null pointer, the function sets the value of pos to the position of the first character in str after the number.
 * @param[in] base Numeric base used to interpret the string.
 *
 * @return Integer representation of @p str.
 */
inline int stoi(const std::string &str, std::size_t *pos = 0, NumericBase base = NumericBase::BASE_10)
{
    assert(base == NumericBase::BASE_10 || base == NumericBase::BASE_16);
    unsigned int      x;
    std::stringstream ss;
    if(base == NumericBase::BASE_16)
    {
        ss << std::hex;
    }
    ss << str;
    ss >> x;
    return x;
}

/** Convert string values to unsigned long.
 *
 * @note This function implements the same behaviour as std::stoul. The latter
 *       is missing in some Android toolchains.
 *
 * @param[in] str  String to be converted to unsigned long.
 * @param[in] pos  If idx is not a null pointer, the function sets the value of pos to the position of the first character in str after the number.
 * @param[in] base Numeric base used to interpret the string.
 *
 * @return Unsigned long representation of @p str.
 */
inline unsigned long stoul(const std::string &str, std::size_t *pos = 0, NumericBase base = NumericBase::BASE_10)
{
    assert(base == NumericBase::BASE_10 || base == NumericBase::BASE_16);
    std::stringstream stream;
    unsigned long     value = 0;
    if(base == NumericBase::BASE_16)
    {
        stream << std::hex;
    }
    stream << str;
    stream >> value;
    return value;
}

#if(__ANDROID__ || BARE_METAL)
/** Convert integer and float values to string.
 *
 * @note This function implements the same behaviour as std::to_string. The
 *       latter is missing in some Android toolchains.
 *
 * @param[in] value Value to be converted to string.
 *
 * @return String representation of @p value.
 */
template <typename T, typename std::enable_if<std::is_arithmetic<typename std::decay<T>::type>::value, int>::type = 0>
inline std::string to_string(T && value)
{
    std::stringstream stream;
    stream << std::forward<T>(value);
    return stream.str();
}

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
    return ::nearbyint(value);
}

/** Convert string values to float.
 *
 * @note This function implements the same behaviour as std::stof. The latter
 *       is missing in some Android toolchains.
 *
 * @param[in] str String to be converted to float.
 *
 * @return Float representation of @p str.
 */
inline float stof(const std::string &str)
{
    std::stringstream stream(str);
    float             value = 0.f;
    stream >> value;
    return value;
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
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T fma(T x, T y, T z)
{
    return ::fma(x, y, z);
}

/** Loads the data from the given location, converts them to character string equivalents
 *  and writes the result to a character string buffer.
 *
 * @param[in] s    Pointer to a character string to write to
 * @param[in] n    Up to buf_size - 1 characters may be written, plus the null terminator
 * @param[in] fmt  Pointer to a null-terminated multibyte string specifying how to interpret the data.
 * @param[in] args Arguments forwarded to snprintf.
 *
 * @return  Number of characters that would have been written for a sufficiently large buffer
 *          if successful (not including the terminating null character), or a negative value if an error occurred.
 */
template <typename... Ts>
inline int snprintf(char *s, size_t n, const char *fmt, Ts &&... args)
{
    return ::snprintf(s, n, fmt, std::forward<Ts>(args)...);
}
#else  /* (__ANDROID__ || BARE_METAL) */
/** Convert integer and float values to string.
 *
 * @note This function acts as a convenience wrapper around std::to_string. The
 *       latter is missing in some Android toolchains.
 *
 * @param[in] value Value to be converted to string.
 *
 * @return String representation of @p value.
 */
template <typename T>
inline std::string to_string(T &&value)
{
    return ::std::to_string(std::forward<T>(value));
}

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
    return std::nearbyint(value);
}

/** Convert string values to float.
 *
 * @note This function acts as a convenience wrapper around std::stof. The
 *       latter is missing in some Android toolchains.
 *
 * @param[in] args Arguments forwarded to std::stof.
 *
 * @return Float representation of input string.
 */
template <typename... Ts>
int stof(Ts &&... args)
{
    return ::std::stof(std::forward<Ts>(args)...);
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
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T fma(T x, T y, T z)
{
    return std::fma(x, y, z);
}

/** Loads the data from the given location, converts them to character string equivalents
 *  and writes the result to a character string buffer.
 *
 * @param[in] s    Pointer to a character string to write to
 * @param[in] n    Up to buf_size - 1 characters may be written, plus the null terminator
 * @param[in] fmt  Pointer to a null-terminated multibyte string specifying how to interpret the data.
 * @param[in] args Arguments forwarded to std::snprintf.
 *
 * @return  Number of characters that would have been written for a sufficiently large buffer
 *          if successful (not including the terminating null character), or a negative value if an error occurred.
 */
template <typename... Ts>
inline int snprintf(char *s, std::size_t n, const char *fmt, Ts &&... args)
{
    return std::snprintf(s, n, fmt, std::forward<Ts>(args)...);
}
#endif /* (__ANDROID__ || BARE_METAL) */

inline std::string to_string(bool value)
{
    std::stringstream str;
    str << std::boolalpha << value;
    return str.str();
}

// std::align is missing in GCC 4.9
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
inline void *align(std::size_t alignment, std::size_t size, void *&ptr, std::size_t &space)
{
    std::uintptr_t pn      = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t aligned = (pn + alignment - 1) & -alignment;
    std::size_t    padding = aligned - pn;
    if(space < size + padding)
    {
        return nullptr;
    }

    space -= padding;

    return ptr = reinterpret_cast<void *>(aligned);
}
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

// std::isfinite
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline bool isfinite(T value)
{
    return std::isfinite(value);
}

inline bool isfinite(half_float::half value)
{
    return half_float::isfinite(value);
}
} // namespace cpp11

namespace cpp14
{
/** make_unique is missing in CPP11. Re-implement it according to the standard proposal. */

/**<Template for single object */
template <class T>
struct _Unique_if
{
    typedef std::unique_ptr<T> _Single_object; /**< Single object type */
};

/** Template for array */
template <class T>
struct _Unique_if<T[]>
{
    typedef std::unique_ptr<T[]> _Unknown_bound; /**< Array type */
};

/** Template for array with known bounds (to throw an error).
 *
 * @note this is intended to never be hit.
 */
template <class T, size_t N>
struct _Unique_if<T[N]>
{
    typedef void _Known_bound; /**< Should never be used */
};

/** Construct a single object and return a unique pointer to it.
 *
 * @param[in] args Constructor arguments.
 *
 * @return a unique pointer to the new object.
 */
template <class T, class... Args>
typename _Unique_if<T>::_Single_object
make_unique(Args &&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/** Construct an array of objects and return a unique pointer to it.
 *
 * @param[in] n Array size
 *
 * @return a unique pointer to the new array.
 */
template <class T>
typename _Unique_if<T>::_Unknown_bound
make_unique(size_t n)
{
    typedef typename std::remove_extent<T>::type U;
    return std::unique_ptr<T>(new U[n]());
}

/** It is invalid to attempt to make_unique an array with known bounds. */
template <class T, class... Args>
typename _Unique_if<T>::_Known_bound
make_unique(Args &&...) = delete;
} // namespace cpp14
} // namespace support
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_TOOLCHAINSUPPORT */
