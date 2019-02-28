/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_MISC_UTILITY_H__
#define __ARM_COMPUTE_MISC_UTILITY_H__

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <vector>

namespace arm_compute
{
namespace utility
{
/** @cond */
template <std::size_t...>
struct index_sequence
{
};

template <std::size_t N, std::size_t... S>
struct index_sequence_generator : index_sequence_generator < N - 1, N - 1, S... >
{
};

template <std::size_t... S>
struct index_sequence_generator<0u, S...> : index_sequence<S...>
{
    using type = index_sequence<S...>;
};

template <std::size_t N>
using index_sequence_t = typename index_sequence_generator<N>::type;
/** @endcond */

namespace detail
{
template <std::size_t... S,
          typename Iterator,
          typename T = std::array<typename std::iterator_traits<Iterator>::value_type, sizeof...(S)>>
T make_array(Iterator first, index_sequence<S...>)
{
    return T{ { first[S]... } };
}
} // namespace detail

template <std::size_t N, typename Iterator>
std::array<typename std::iterator_traits<Iterator>::value_type, N> make_array(Iterator first, Iterator last)
{
    return detail::make_array(first, index_sequence_t<N> {});
}

/** Performs clamping among a lower and upper value.
 *
 * @param[in] n     Value to clamp.
 * @param[in] lower Lower threshold.
 * @param[in] upper Upper threshold.
 *
 *  @return Clamped value.
 */
template <typename DataType, typename RangeType = DataType>
inline DataType clamp(const DataType &n,
                      const DataType &lower = std::numeric_limits<RangeType>::lowest(),
                      const DataType &upper = std::numeric_limits<RangeType>::max())
{
    return std::max(lower, std::min(n, upper));
}

/** Base case of for_each. Does nothing. */
template <typename F>
inline void for_each(F &&)
{
}

/** Call the function for each of the arguments
 *
 * @param[in] func Function to be called
 * @param[in] arg  Argument passed to the function
 * @param[in] args Remaining arguments
 */
template <typename F, typename T, typename... Ts>
inline void for_each(F &&func, T &&arg, Ts &&... args)
{
    func(std::forward<T>(arg));
    for_each(std::forward<F>(func), std::forward<Ts>(args)...);
}

/** Base case of foldl.
 *
 * @return value.
 */
template <typename F, typename T>
inline T &&foldl(F &&, T &&value)
{
    return std::forward<T>(value);
}

/** Fold left.
 *
 * @param[in] func    Function to be called
 * @param[in] initial Initial value
 * @param[in] value   Argument passed to the function
 * @param[in] values  Remaining arguments
 */
template <typename F, typename T, typename U, typename... Us>
inline auto foldl(F &&func, T &&initial, U &&value, Us &&... values) -> decltype(func(std::forward<T>(initial), std::forward<U>(value)))
{
    return foldl(std::forward<F>(func), func(std::forward<T>(initial), std::forward<U>(value)), std::forward<Us>(values)...);
}

/** Perform an index sort of a given vector.
 *
 * @param[in] v Vector to sort
 *
 * @return Sorted index vector.
 */
template <typename T>
std::vector<size_t> sort_indices(const std::vector<T> &v)
{
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2)
    {
        return v[i1] < v[i2];
    });

    return idx;
}

/** Checks if a string contains a given suffix
 *
 * @param[in] str    Input string
 * @param[in] suffix Suffix to check for
 *
 * @return True if the string ends with the given suffix else false
 */
inline bool endswith(const std::string &str, const std::string &suffix)
{
    if(str.size() < suffix.size())
    {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

/** Checks if a pointer complies with a given alignment
 *
 * @param[in] ptr       Pointer to check
 * @param[in] alignment Alignment value
 *
 * @return True if the pointer is aligned else false
 */
inline bool check_aligned(void *ptr, const size_t alignment)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
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
} // namespace utility
} // namespace arm_compute
#endif /* __ARM_COMPUTE_MISC_UTILITY_H__ */
