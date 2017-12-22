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
#ifndef ARM_COMPUTE_TEST_PRINTER
#define ARM_COMPUTE_TEST_PRINTER

#include "tests/framework/Profiler.h"

#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>

namespace arm_compute
{
namespace test
{
namespace framework
{
struct TestInfo;

/** Abstract printer class used by the @ref Framework to present output. */
class Printer
{
public:
    /** Default constructor.
     *
     * Prints values to std::cout.
     * */
    Printer() = default;

    /** Construct printer with given output stream.
     *
     * @param[out] stream Output stream.
     */
    Printer(std::ostream &stream);

    Printer(const Printer &) = delete;
    Printer &operator=(const Printer &) = delete;
    Printer(Printer &&)                 = default;
    Printer &operator=(Printer &&) = default;

    virtual ~Printer() = default;

    /** Print given string.
     *
     * @param[in] str String.
     */
    void print(const std::string &str);

    /** Print an entry consisting of a (name, value) pair.
     *
     * @param[in] name  Description of the value.
     * @param[in] value Value.
     */
    virtual void print_entry(const std::string &name, const std::string &value) = 0;

    /** Print global header. */
    virtual void print_global_header() = 0;

    /** Print global footer. */
    virtual void print_global_footer() = 0;

    /** Print header before running all tests. */
    virtual void print_run_header() = 0;

    /** Print footer after running all tests. */
    virtual void print_run_footer() = 0;

    /** Print header before a test.
     *
     * @param[in] info Test info.
     */
    virtual void print_test_header(const TestInfo &info) = 0;

    /** Print footer after a test. */
    virtual void print_test_footer() = 0;

    /** Print header before errors. */
    virtual void print_errors_header() = 0;

    /** Print footer after errors. */
    virtual void print_errors_footer() = 0;

    /** Print the list of all the tests
     *
     * @param[in] infos List of tests to print
     */
    virtual void print_list_tests(const std::vector<TestInfo> &infos) = 0;
    /** Print test error.
     *
     * @param[in] error    Description of the error.
     * @param[in] expected Whether the error was expected or not.
     */
    virtual void print_error(const std::exception &error, bool expected) = 0;

    /** Print test log info.
     *
     * @param[in] info Description of the log.
     */
    virtual void print_info(const std::string &info) = 0;

    /** Print measurements for a test.
     *
     * @param[in] measurements Measurements as collected by a @ref Profiler.
     */
    virtual void print_measurements(const Profiler::MeasurementsMap &measurements) = 0;

    /** Set the output stream.
     *
     * @param[out] stream Output stream.
     */
    void set_stream(std::ostream &stream);

protected:
    std::ostream *_stream{ &std::cout };
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PRINTER */
