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
#include "arm_compute/core/utils/logging/Logger.h"

#include "arm_compute/core/Error.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::logging;

Logger::Logger(std::string name, LogLevel log_level, std::shared_ptr<Printer> printer)
    : _name(std::move(name)), _log_level(log_level), _printers(
{
    std::move(printer)
}), _decorators()
{
    // Check printer
    ARM_COMPUTE_ERROR_ON(printer == nullptr);

    // Set default message decorators
    set_default_decorators();
}

Logger::Logger(std::string name, LogLevel log_level, std::vector<std::shared_ptr<Printer>> printers)
    : _name(std::move(name)), _log_level(log_level), _printers(std::move(printers)), _decorators()
{
    // Check printers
    for(const auto &p : _printers)
    {
        ARM_COMPUTE_UNUSED(p);
        ARM_COMPUTE_ERROR_ON(p == nullptr);
    }
    // Set default message decorators
    set_default_decorators();
}

Logger::Logger(std::string                              name,
               LogLevel                                 log_level,
               std::vector<std::shared_ptr<Printer>>    printers,
               std::vector<std::unique_ptr<IDecorator>> decorators)
    : _name(std::move(name)), _log_level(log_level), _printers(std::move(printers)), _decorators(std::move(decorators))
{
    // Check printers
    for(const auto &p : _printers)
    {
        ARM_COMPUTE_UNUSED(p);
        ARM_COMPUTE_ERROR_ON(p == nullptr);
    }
    // Check decorators
    for(const auto &d : _decorators)
    {
        ARM_COMPUTE_UNUSED(d);
        ARM_COMPUTE_ERROR_ON(d == nullptr);
    }
}

void Logger::log(LogLevel log_level, const std::string &msg)
{
    // Return if message shouldn't be logged
    // i.e. if log level does not match the logger's
    if(!is_loggable(log_level))
    {
        return;
    }

    // Print message to all printers
    print_all(create_log_msg(msg, log_level));
}

void Logger::set_log_level(LogLevel log_level)
{
    _log_level = log_level;
}

LogLevel Logger::log_level() const
{
    return _log_level;
}

std::string Logger::name() const
{
    return _name;
}

void Logger::add_printer(std::shared_ptr<Printer> printer)
{
    ARM_COMPUTE_ERROR_ON(printer == nullptr);
    _printers.push_back(std::move(printer));
}

void Logger::add_decorator(std::unique_ptr<IDecorator> decorator)
{
    ARM_COMPUTE_ERROR_ON(decorator == nullptr);
    _decorators.push_back(std::move(decorator));
}

void Logger::set_default_decorators()
{
    _decorators.emplace_back(support::cpp14::make_unique<StringDecorator>(_name));
    _decorators.emplace_back(support::cpp14::make_unique<DateDecorator>());
    _decorators.emplace_back(support::cpp14::make_unique<LogLevelDecorator>());
}

bool Logger::is_loggable(LogLevel log_level)
{
    return (log_level >= _log_level);
}

void Logger::decorate_log_msg(LogMsg &msg)
{
    for(const auto &d : _decorators)
    {
        d->decorate(msg);
    }
    msg.raw_ += std::string(" ");
}

std::string Logger::create_log_msg(const std::string &str, LogLevel log_level)
{
    // Adding space string to avoid Android failures
    LogMsg log_msg(" ", log_level);
    decorate_log_msg(log_msg);
    std::ostringstream ss;
    ss << log_msg.raw_ << " " << str;
    return ss.str();
}

void Logger::print_all(const std::string &msg)
{
    for(auto &p : _printers)
    {
        p->print(msg);
    }
}