/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_COMMON_UTILS_H
#define SRC_COMMON_UTILS_H

#include <type_traits>

namespace arm_compute
{
namespace utils
{
/** Convert a strongly typed enum to an old plain c enum
 *
 * @tparam E  Plain old C enum
 * @tparam SE Strongly typed resulting enum
 *
 * @param[in] v Value to convert
 *
 * @return A corresponding plain old C enumeration
 */
template <typename E, typename SE>
constexpr E as_cenum(const SE v) noexcept
{
    return static_cast<E>(static_cast<std::underlying_type_t<SE>>(v));
}

/** Convert plain old enumeration to a strongly typed enum
 *
 * @tparam SE Strongly typed resulting enum
 * @tparam E  Plain old C enum
 *
 * @param[in] val Value to convert
 *
 * @return A corresponding strongly typed enumeration
 */
template <typename SE, typename E>
constexpr SE as_enum(const E val) noexcept
{
    return static_cast<SE>(val);
}

/** Check if the given value is in the given enum value list
 *
 * @tparam E  The type of the enum
 *
 * @param[in] check Value to check
 * @param[in] list  List of enum values to check against
 *
 * @return True if the given value is found in the list
 */
template <typename E>
bool is_in(E check, std::initializer_list<E> list)
{
    return std::any_of(std::cbegin(list), std::cend(list), [&check](E e)
    {
        return check == e;
    });
}
} // namespace utils
} // namespace arm_compute

#endif /* SRC_COMMON_UTILS_H */
