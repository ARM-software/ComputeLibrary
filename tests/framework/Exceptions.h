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
#ifndef ARM_COMPUTE_TEST_EXCEPTIONS
#define ARM_COMPUTE_TEST_EXCEPTIONS

#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Severity of the information.
 *
 * Each category includes the ones above it.
 *
 * NONE == Only for filtering. Not used to tag information.
 * CONFIG == Configuration info.
 * TESTS == Information about the tests.
 * ERRORS == Violated assertions/expectations.
 * DEBUG == More violated assertions/expectations.
 * MEASUREMENTS == Information about measurements.
 * ALL == Only for filtering. Not used to tag information.
 */
enum class LogLevel
{
    NONE,
    CONFIG,
    TESTS,
    ERRORS,
    DEBUG,
    MEASUREMENTS,
    ALL,
};

LogLevel log_level_from_name(const std::string &name);
::std::istream &operator>>(::std::istream &stream, LogLevel &level);
::std::ostream &operator<<(::std::ostream &stream, LogLevel level);
std::string to_string(LogLevel level);

/** Error class for when some external assets are missing */
class FileNotFound : public std::runtime_error
{
public:
    /** Construct error with message
     *
     * @param[in] msg Error message
     */
    FileNotFound(const std::string &msg);
};

/** Error class for failures during test execution. */
class TestError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;

    /** Construct error with severity.
     *
     * @param[in] msg     Error message.
     * @param[in] level   Severity level.
     * @param[in] context Context.
     */
    TestError(const std::string &msg, LogLevel level, std::string context = "");

    /** Severity of the error.
     *
     * @return Severity.
     */
    LogLevel level() const;

    const char *what() const noexcept override;

private:
    LogLevel    _level{ LogLevel::ERRORS };
    std::string _msg{};
    std::string _context{};
    std::string _combined{};
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_EXCEPTIONS */
