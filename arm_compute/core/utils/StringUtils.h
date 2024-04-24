/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CORE_UTILS_STRINGUTILS_H
#define ARM_COMPUTE_CORE_UTILS_STRINGUTILS_H

#include <string>
#include <vector>

namespace arm_compute
{
/** Lower a given string.
 *
 * @param[in] val Given string to lower.
 *
 * @return The lowered string
 */
std::string lower_string(const std::string &val);

/** Raise a given string to upper case
 *
 * @param[in] val Given string to lower.
 *
 * @return The upper case string
 */
std::string upper_string(const std::string &val);

/** Create a string with the float in full precision.
 *
 * @param val Floating point value
 *
 * @return String with the floating point value.
 */
std::string float_to_string_with_full_precision(float val);

/** Join a sequence of strings with separator @p sep
 *
 * @param[in] strings Strings to join
 * @param[in] sep     Separator to join consecutive strings in the sequence
 *
 * @return std::string
 */
std::string join(const std::vector<std::string> strings, const std::string &sep);
} // namespace arm_compute
#endif /*ARM_COMPUTE_CORE_UTILS_STRINGUTILS_H */
