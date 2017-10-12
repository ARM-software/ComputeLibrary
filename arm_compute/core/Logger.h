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

#ifndef __ARM_COMPUTE_LOGGER_H__
#define __ARM_COMPUTE_LOGGER_H__

#include <iostream>
#include <memory>

#ifdef ARM_COMPUTE_DEBUG_ENABLED
#define ARM_COMPUTE_LOG(x) (arm_compute::Logger::get().log_info() << x)
#else /* ARM_COMPUTE_DEBUG_ENABLED */
#define ARM_COMPUTE_LOG(...)
#endif /* ARM_COMPUTE_DEBUG_ENABLED */

namespace arm_compute
{
/**< Verbosity of the logger */
enum class LoggerVerbosity
{
    NONE, /**< No info */
    INFO  /**< Log info */
};

/** Logger singleton class */
class Logger
{
public:
    static Logger &get();
    void set_logger(std::ostream &ostream, LoggerVerbosity verbosity);
    std::ostream &log_info();

private:
    /** Default constructor */
    Logger();
    /** Allow instances of this class to be moved */
    Logger(Logger &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Logger(const Logger &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Logger &operator=(const Logger &) = delete;
    /** Allow instances of this class to be moved */
    Logger &operator=(Logger &&) = default;

    std::ostream   *_ostream;
    std::ostream    _nullstream;
    LoggerVerbosity _verbosity;
};
} // arm_compute
#endif /* __ARM_COMPUTE_LOGGER_H__ */