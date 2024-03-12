/*
* Copyright (c) 2020 Arm Limited.
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
#ifndef SUPPORT_TRAITS_H
#define SUPPORT_TRAITS_H

#include <type_traits>

namespace arm_compute
{
/** Disable bitwise operations by default */
template <typename T>
struct enable_bitwise_ops
{
    static constexpr bool value = false; /**< Disabled */
};

#ifndef DOXYGEN_SKIP_THIS
template <typename T>
typename std::enable_if<enable_bitwise_ops<T>::value, T>::type operator&(T lhs, T rhs)
{
    using underlying_type = typename std::underlying_type<T>::type;
    return static_cast<T>(static_cast<underlying_type>(lhs) & static_cast<underlying_type>(rhs));
}
#endif /* DOXYGEN_SKIP_THIS */
} // namespace arm_compute
#endif /* SUPPORT_TRAITS_H */
