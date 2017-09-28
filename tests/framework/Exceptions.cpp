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
#include "Exceptions.h"

#include "Utils.h"

#include <map>
#include <sstream>

namespace arm_compute
{
namespace test
{
namespace framework
{
LogLevel log_level_from_name(const std::string &name)
{
    static const std::map<std::string, LogLevel> levels =
    {
        { "none", LogLevel::NONE },
        { "config", LogLevel::CONFIG },
        { "tests", LogLevel::TESTS },
        { "errors", LogLevel::ERRORS },
        { "debug", LogLevel::DEBUG },
        { "measurements", LogLevel::MEASUREMENTS },
        { "all", LogLevel::ALL },
    };

    try
    {
        return levels.at(tolower(name));
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
}

::std::istream &operator>>(::std::istream &stream, LogLevel &level)
{
    std::string value;
    stream >> value;
    level = log_level_from_name(value);
    return stream;
}

::std::ostream &operator<<(::std::ostream &stream, LogLevel level)
{
    switch(level)
    {
        case LogLevel::NONE:
            stream << "NONE";
            break;
        case LogLevel::CONFIG:
            stream << "CONFIG";
            break;
        case LogLevel::TESTS:
            stream << "TESTS";
            break;
        case LogLevel::ERRORS:
            stream << "ERRORS";
            break;
        case LogLevel::DEBUG:
            stream << "DEBUG";
            break;
        case LogLevel::MEASUREMENTS:
            stream << "MEASUREMENTS";
            break;
        case LogLevel::ALL:
            stream << "ALL";
            break;
        default:
            throw std::invalid_argument("Unsupported log level");
    }

    return stream;
}

std::string to_string(LogLevel level)
{
    std::stringstream stream;
    stream << level;
    return stream.str();
}

FileNotFound::FileNotFound(const std::string &msg)
    : std::runtime_error{ msg }
{
}

TestError::TestError(const std::string &msg, LogLevel level, std::string context)
    : std::runtime_error{ msg }, _level{ level }, _msg{ msg }, _context{ std::move(context) }, _combined{ "ERROR: " + msg }
{
    if(!_context.empty())
    {
        _combined += "\nCONTEXT:\n" + _context;
    }
}

LogLevel TestError::level() const
{
    return _level;
}

const char *TestError::what() const noexcept
{
    return _combined.c_str();
}

} // namespace framework
} // namespace test
} // namespace arm_compute
