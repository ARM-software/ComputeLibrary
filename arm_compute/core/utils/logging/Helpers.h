/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_LOGGING_HELPERS_H
#define ARM_COMPUTE_LOGGING_HELPERS_H

#include "arm_compute/core/utils/logging/Types.h"
#include "support/ToolchainSupport.h"

#include <cstddef>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

namespace arm_compute
{
namespace logging
{
/** Create a string given a format
 *
 * @param[in] fmt  String format
 * @param[in] args Arguments
 *
 * @return The formatted string
 */
template <typename... Ts>
inline std::string string_with_format(const std::string &fmt, Ts &&... args)
{
    size_t size     = support::cpp11::snprintf(nullptr, 0, fmt.c_str(), args...) + 1;
    auto   char_str = std::make_unique<char[]>(size);
    support::cpp11::snprintf(char_str.get(), size, fmt.c_str(), args...);
    return std::string(char_str.get(), char_str.get() + size - 1);
}
/** Wraps a value with angles and returns the string
 *
 * @param[in] val Value to wrap
 *
 * @return Wrapped string
 */
template <typename T>
inline std::string angle_wrap_value(const T &val)
{
    std::ostringstream ss;
    ss << "[" << val << "]";
    return ss.str();
}
/** Translates a given log level to a string.
 *
 * @param[in] log_level @ref LogLevel to be translated to string.
 *
 * @return The string describing the logging level.
 */
const std::string &string_from_log_level(LogLevel log_level);
} // namespace logging
} // namespace arm_compute
#endif /* ARM_COMPUTE_LOGGING_HELPERS_H */
