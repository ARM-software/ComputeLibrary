/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_LOGGING_LOGGER_H
#define ARM_COMPUTE_LOGGING_LOGGER_H

#include "arm_compute/core/utils/logging/Helpers.h"
#include "arm_compute/core/utils/logging/IPrinter.h"
#include "arm_compute/core/utils/logging/LogMsgDecorators.h"
#include "arm_compute/core/utils/logging/Types.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace arm_compute
{
namespace logging
{
/** Logger class */
class Logger
{
public:
    /** Default Constructor
     *
     * @param[in] name      Name of the logger
     * @param[in] log_level Logger log level
     * @param[in] printer   Printer to push the messages
     */
    Logger(std::string name, LogLevel log_level, std::shared_ptr<Printer> printer);
    /** Default Constructor
     *
     * @param[in] name      Name of the logger
     * @param[in] log_level Logger log level
     * @param[in] printers  Printers to push the messages
     */
    Logger(std::string name, LogLevel log_level, std::vector<std::shared_ptr<Printer>> printers = {});
    /** Default Constructor
     *
     * @param[in] name       Name of the logger
     * @param[in] log_level  Logger log level
     * @param[in] printers   Printers to push the messages
     * @param[in] decorators Message decorators, which append information in the logged message
     */
    Logger(std::string                              name,
           LogLevel                                 log_level,
           std::vector<std::shared_ptr<Printer>>    printers,
           std::vector<std::unique_ptr<IDecorator>> decorators);
    /** Allow instances of this class to be moved */
    Logger(Logger &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Logger(const Logger &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Logger &operator=(const Logger &) = delete;
    /** Allow instances of this class to be moved */
    Logger &operator=(Logger &&) = default;
    /** Logs a message
     *
     * @param[in] log_level Log level of the message
     * @param[in] msg       Message to log
     */
    void log(LogLevel log_level, const std::string &msg);
    /** Logs a formatted message
     *
     * @param[in] log_level Log level of the message
     * @param[in] fmt       Message format
     * @param[in] args      Message arguments
     */
    template <typename... Ts>
    void log(LogLevel log_level, const std::string &fmt, Ts &&... args);
    /** Sets log level of the logger
     *
     * @warning Not thread-safe
     *
     * @param[in] log_level Log level to set
     */
    void set_log_level(LogLevel log_level);
    /** Returns logger's log level
     *
     * @return Logger's log level
     */
    LogLevel log_level() const;
    /** Returns logger's name
     *
     * @return Logger's name
     */
    std::string name() const;
    /** Adds a printer to the logger
     *
     * @warning Not thread-safe
     *
     * @param[in] printer
     */
    void add_printer(std::shared_ptr<Printer> printer);
    /** Adds a log message decorator to the logger
     *
     * @warning Not thread-safe
     *
     * @param[in] decorator
     */
    void add_decorator(std::unique_ptr<IDecorator> decorator);

private:
    /** Set default message decorators */
    void set_default_decorators();
    /** Checks if a message should be logged depending
     *  on the message log level and the loggers one
     *
     * @param[in] log_level Log level
     *
     * @return True if message should be logged else false
     */
    bool is_loggable(LogLevel log_level);
    /** Decorate log message
     *
     * @param[in] Log message to decorate
     */
    void decorate_log_msg(LogMsg &msg);
    /** Creates final log message by creating the prefix
     *
     * @param[in] str       Log message
     * @param[in] log_level Message's log level
     *
     * @return Final log message to print
     */
    std::string create_log_msg(const std::string &str, LogLevel log_level);
    /** Prints the message to all the printers
     *
     * @param[in] msg Message to print
     */
    void print_all(const std::string &msg);

private:
    std::string                              _name;
    LogLevel                                 _log_level;
    std::vector<std::shared_ptr<Printer>>    _printers;
    std::vector<std::unique_ptr<IDecorator>> _decorators;
};

template <typename... Ts>
inline void Logger::log(LogLevel log_level, const std::string &fmt, Ts &&... args)
{
    // Return if message shouldn't be logged
    // i.e. if log level does not match the logger's
    if(!is_loggable(log_level))
    {
        return;
    }

    // Print message to all printers
    print_all(create_log_msg(string_with_format(fmt, args...), log_level));
}
} // namespace logging
} // namespace arm_compute
#endif /* ARM_COMPUTE_LOGGING_LOGGER_H */
