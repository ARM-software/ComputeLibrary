/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_LOGGING_LOG_MSG_DECORATORS_H__
#define __ARM_COMPUTE_LOGGING_LOG_MSG_DECORATORS_H__

#include "arm_compute/core/utils/logging/Helpers.h"
#include "arm_compute/core/utils/logging/Types.h"

#include <chrono>
#include <ctime>
#include <string>
#include <thread>

namespace arm_compute
{
namespace logging
{
/** Log message decorator interface */
class IDecorator
{
public:
    /** Default Destructor */
    virtual ~IDecorator() = default;
    /** Decorates log message
     *
     * @param[in] log_msg Log message to decorate
     */
    virtual void decorate(LogMsg &log_msg) = 0;
};

/** String Decorator
 *
 * Appends a user defined string in the log message
 */
class StringDecorator : public IDecorator
{
public:
    /** Defaults constructor
     *
     * @param str Sting to append
     */
    StringDecorator(const std::string &str)
        : _str(str)
    {
        _str = angle_wrap_value(str);
    }

    // Inherited methods overridden:
    void decorate(LogMsg &log_msg) override
    {
        log_msg.raw_ += _str;
    }

private:
    std::string _str;
};

/** Date Decorator
 *
 * Appends the date and time in the log message
 */
class DateDecorator : public IDecorator
{
public:
    // Inherited methods overridden:
    void decorate(LogMsg &log_msg) override
    {
        log_msg.raw_ += angle_wrap_value(get_time());
    }

private:
    /** Gets current system local time
     *
     * @return Local time
     */
    std::string get_time()
    {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        char buf[100] = { 0 };
        std::strftime(buf, sizeof(buf), "%d-%m-%Y %I:%M:%S", std::localtime(&time));
        return buf;
    }
};

/** Thread ID Decorator
 *
 * Appends the thread ID in the log message
 */
class ThreadIdDecorator : public IDecorator
{
public:
    // Inherited methods overridden:
    void decorate(LogMsg &log_msg) override
    {
#ifndef NO_MULTI_THREADING
        log_msg.raw_ += angle_wrap_value(std::this_thread::get_id());
#endif /* NO_MULTI_THREADING */
    }
};

/** Log Level Decorator
 *
 * Appends the logging level in the log message
 */
class LogLevelDecorator : public IDecorator
{
public:
    // Inherited methods overridden:
    void decorate(LogMsg &log_msg) override
    {
        log_msg.raw_ += angle_wrap_value(string_from_log_level(log_msg.log_level_));
    }
};
} // namespace logging
} // namespace arm_compute
#endif /* __ARM_COMPUTE_LOGGING_LOG_MSG_DECORATORS_H__ */
