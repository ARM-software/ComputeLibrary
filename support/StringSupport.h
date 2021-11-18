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
#ifndef ARM_COMPUTE_TEST_STRINGSUPPORT
#define ARM_COMPUTE_TEST_STRINGSUPPORT

#include <cassert>
#include <memory>
#include <sstream>
#include <string>

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

    if(pos)
    {
        std::string       s;
        std::stringstream ss_p;

        ss_p << x;
        ss_p >> s;
        *pos = s.length();
    }

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

    if(pos)
    {
        std::string       s;
        std::stringstream ss_p;

        ss_p << value;
        ss_p >> s;
        *pos = s.length();
    }

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

// Specialization for const std::string&
inline std::string to_string(const std::string &value)
{
    return value;
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

#else /* (__ANDROID__ || BARE_METAL) */
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

// Specialization for const std::string&
inline std::string to_string(const std::string &value)
{
    return value;
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

#endif /* (__ANDROID__ || BARE_METAL) */

inline std::string to_string(bool value)
{
    std::stringstream str;
    str << std::boolalpha << value;
    return str.str();
}

} // namespace cpp11
} // namespace support
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_STRINGSUPPORT */
