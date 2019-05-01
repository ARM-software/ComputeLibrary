/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/utils/logging/LoggerRegistry.h"

#include "arm_compute/core/Error.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::logging;

/** Reserved logger used by the library */
std::set<std::string> LoggerRegistry::_reserved_loggers = { "CORE", "RUNTIME", "GRAPH" };

LoggerRegistry::LoggerRegistry()
    : _mtx(), _loggers()
{
}

LoggerRegistry &LoggerRegistry::get()
{
    static LoggerRegistry _instance;
    return _instance;
}

void LoggerRegistry::create_logger(const std::string &name, LogLevel log_level, const std::vector<std::shared_ptr<Printer>> &printers)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);
    if((_loggers.find(name) == _loggers.end()) && (_reserved_loggers.find(name) == _reserved_loggers.end()))
    {
        _loggers[name] = std::make_shared<Logger>(name, log_level, printers);
    }
}

void LoggerRegistry::remove_logger(const std::string &name)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);
    if(_loggers.find(name) != _loggers.end())
    {
        _loggers.erase(name);
    }
}

std::shared_ptr<Logger> LoggerRegistry::logger(const std::string &name)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);
    return (_loggers.find(name) != _loggers.end()) ? _loggers[name] : nullptr;
}

void LoggerRegistry::create_reserved_loggers(LogLevel log_level, const std::vector<std::shared_ptr<Printer>> &printers)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);
    for(const auto &r : _reserved_loggers)
    {
        if(_loggers.find(r) == _loggers.end())
        {
            _loggers[r] = std::make_shared<Logger>(r, log_level, printers);
        }
    }
}
