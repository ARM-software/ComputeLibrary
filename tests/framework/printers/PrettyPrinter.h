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
#ifndef ARM_COMPUTE_TEST_PRETTYPRINTER
#define ARM_COMPUTE_TEST_PRETTYPRINTER

#include "Printer.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Implementation of a @ref Printer that produces human readable output. */
class PrettyPrinter : public Printer
{
public:
    using Printer::Printer;

    /** Set if the output is colored.
     *
     * @param[in] color_output True if the output is colored.
     */
    void set_color_output(bool color_output);

    void print_entry(const std::string &name, const std::string &value) override;
    void print_global_header() override;
    void print_global_footer() override;
    void print_run_header() override;
    void print_run_footer() override;
    void print_test_header(const TestInfo &info) override;
    void print_test_footer() override;
    void print_errors_header() override;
    void print_errors_footer() override;
    void print_error(const std::exception &error, bool expected) override;
    void print_info(const std::string &info) override;
    void print_measurements(const Profiler::MeasurementsMap &measurements) override;
    void print_list_tests(const std::vector<TestInfo> &infos) override;

private:
    std::string begin_color(const std::string &color) const;
    std::string end_color() const;

    bool _color_output{ true };
};
} // namespace framework
} // namespace test
} // namespace arm_compute

#endif /* ARM_COMPUTE_TEST_PRETTYPRINTER */
