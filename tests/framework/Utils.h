/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_UTILS
#define ARM_COMPUTE_TEST_UTILS

#include "support/ToolchainSupport.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** @cond */
namespace detail
{
template <int...>
struct sequence
{
};

template <int N, int... Ns>
struct sequence_generator;

template <int... Ns>
struct sequence_generator<0, Ns...>
{
    using type = sequence<Ns...>;
};

template <int N, int... Ns>
struct sequence_generator : sequence_generator < N - 1, N - 1, Ns... >
{
};

template <int N>
using sequence_t = typename sequence_generator<N>::type;
/** @endcond */

template <typename O, typename F, typename... As, int... S>
void apply_impl(O *obj, F &&func, const std::tuple<As...> &args, detail::sequence<S...>)
{
    (obj->*func)(std::get<S>(args)...);
}
} // namespace

template <typename O, typename F, typename... As>
void apply(O *obj, F &&func, const std::tuple<As...> &args)
{
    detail::apply_impl(obj, std::forward<F>(func), args, detail::sequence_t<sizeof...(As)>());
}

/** Helper function to concatenate multiple strings.
 *
 * @param[in] first     Iterator pointing to the first element to be concatenated.
 * @param[in] last      Iterator pointing behind the last element to be concatenated.
 * @param[in] separator String used to join the elements.
 *
 * @return String containing all elements joined by @p separator.
 */
template <typename T, typename std::enable_if<std::is_same<typename T::value_type, std::string>::value, int>::type = 0>
std::string join(T first, T last, const std::string &separator)
{
    return std::accumulate(std::next(first), last, *first, [&separator](const std::string & base, const std::string & suffix)
    {
        return base + separator + suffix;
    });
}

/** Helper function to concatenate multiple values.
 *
 * All values are converted to std::string using the provided operation before
 * being joined.
 *
 * The signature of op has to be equivalent to
 * std::string op(const T::value_type &val).
 *
 * @param[in] first     Iterator pointing to the first element to be concatenated.
 * @param[in] last      Iterator pointing behind the last element to be concatenated.
 * @param[in] separator String used to join the elements.
 * @param[in] op        Conversion function.
 *
 * @return String containing all elements joined by @p separator.
 */
template <typename T, typename UnaryOp>
std::string join(T &&first, T &&last, const std::string &separator, UnaryOp &&op)
{
    return std::accumulate(std::next(first), last, op(*first), [&separator, &op](const std::string & base, const typename T::value_type & suffix)
    {
        return base + separator + op(suffix);
    });
}

/** Helper function to concatenate multiple values.
 *
 * All values are converted to std::string using std::to_string before being joined.
 *
 * @param[in] first     Iterator pointing to the first element to be concatenated.
 * @param[in] last      Iterator pointing behind the last element to be concatenated.
 * @param[in] separator String used to join the elements.
 *
 * @return String containing all elements joined by @p separator.
 */
template <typename T, typename std::enable_if<std::is_arithmetic<typename T::value_type>::value, int>::type = 0>
std::string join(T && first, T && last, const std::string &separator)
{
    return join(std::forward<T>(first), std::forward<T>(last), separator, support::cpp11::to_string);
}

/** Convert string to lower case.
 *
 * @param[in] string To be converted string.
 *
 * @return Lower case string.
 */
inline std::string tolower(std::string string)
{
    std::transform(string.begin(), string.end(), string.begin(), [](unsigned char c)
    {
        return std::tolower(c);
    });
    return string;
}

/** Create a string with the arithmetic value in full precision.
 *
 * @param val            Arithmetic value
 * @param decimal_places How many decimal places to show
 *
 * @return String with the arithmetic value.
 */
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
inline std::string arithmetic_to_string(T val, int decimal_places = 0)
{
    std::stringstream ss;
    ss << std::fixed;
    ss.precision((decimal_places) ? decimal_places : std::numeric_limits<T>::digits10 + 1);
    ss << val;
    return ss.str();
}

} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UTILS */
