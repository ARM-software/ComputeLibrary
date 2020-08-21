/*
 * Copyright (c) 2016-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_LOGGING_TYPES_H
#define ARM_COMPUTE_LOGGING_TYPES_H

#include <string>

namespace arm_compute
{
namespace logging
{
/** Logging level enumeration */
enum class LogLevel
{
    VERBOSE, /**< All logging messages */
    INFO,    /**< Information log level */
    WARN,    /**< Warning log level */
    ERROR,   /**< Error log level */
    OFF      /**< No logging */
};

/** Log message */
struct LogMsg
{
    /** Default constructor */
    LogMsg()
        : raw_(), log_level_(LogLevel::OFF)
    {
    }
    /** Construct a log message
     *
     * @param[in] msg       Message to log.
     * @param[in] log_level Logging level. Default: OFF
     */
    LogMsg(std::string msg, LogLevel log_level = LogLevel::OFF)
        : raw_(msg), log_level_(log_level)
    {
    }

    /** Log message */
    std::string raw_;
    /** Logging level */
    LogLevel log_level_;
};
} // namespace logging
} // namespace arm_compute
#endif /* ARM_COMPUTE_TYPES_H */
