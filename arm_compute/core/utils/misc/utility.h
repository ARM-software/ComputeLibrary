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
#ifndef __ARM_COMPUTE_MISC_UTILITY_H__
#define __ARM_COMPUTE_MISC_UTILITY_H__

#include <array>

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
} // namespace misc
} // namespace arm_compute
#endif /* __ARM_COMPUTE_MISC_UTILITY_H__ */
