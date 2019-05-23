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
#include "PrettyPrinter.h"

#include "../Framework.h"
#include "../instruments/InstrumentsStats.h"
#include "../instruments/Measurement.h"

#include <algorithm>

namespace arm_compute
{
namespace test
{
namespace framework
{
std::string PrettyPrinter::begin_color(const std::string &color) const
{
    if(!_color_output)
    {
        return "";
    }

    return "\033[0;3" + color + "m";
}

std::string PrettyPrinter::end_color() const
{
    if(!_color_output)
    {
        return "";
    }

    return "\033[m";
}

void PrettyPrinter::set_color_output(bool color_output)
{
    _color_output = color_output;
}

void PrettyPrinter::print_entry(const std::string &name, const std::string &value)
{
    *_stream << begin_color("4") << name << " = " << value << end_color() << "\n";
}

void PrettyPrinter::print_global_header()
{
}

void PrettyPrinter::print_global_footer()
{
}

void PrettyPrinter::print_run_header()
{
}

void PrettyPrinter::print_run_footer()
{
}

void PrettyPrinter::print_test_header(const TestInfo &info)
{
    *_stream << begin_color("2") << "Running [" << info.id << "] '" << info.name << "'" << end_color() << "\n";
}

void PrettyPrinter::print_test_footer()
{
}

void PrettyPrinter::print_errors_header()
{
}

void PrettyPrinter::print_errors_footer()
{
}

void PrettyPrinter::print_info(const std::string &info)
{
    *_stream << begin_color("1") << "INFO: " << info << end_color() << "\n";
}

void PrettyPrinter::print_error(const std::exception &error, bool expected)
{
    std::string prefix = expected ? "EXPECTED ERROR: " : "ERROR: ";
    *_stream << begin_color("1") << prefix << error.what() << end_color() << "\n";
}

void PrettyPrinter::print_list_tests(const std::vector<TestInfo> &infos)
{
    for(auto const &info : infos)
    {
        *_stream << "[" << info.id << ", " << info.mode << ", " << info.status << "] " << info.name << "\n";
    }
}
void PrettyPrinter::print_measurements(const Profiler::MeasurementsMap &measurements)
{
    for(const auto &instrument : measurements)
    {
        *_stream << begin_color("3") << "  " << instrument.first << ":";

        InstrumentsStats stats(instrument.second);

        *_stream << "    ";
        *_stream << "AVG=" << stats.mean() << " " << stats.max().unit();
        if(instrument.second.size() > 1)
        {
            *_stream << ", STDDEV=" << arithmetic_to_string(stats.relative_standard_deviation(), 2) << " %";
            *_stream << ", MIN=" << stats.min();
            *_stream << ", MAX=" << stats.max();
            *_stream << ", MEDIAN=" << stats.median().value() << " " << stats.median().unit();
        }
        *_stream << end_color() << "\n";
    }
}
} // namespace framework
} // namespace test
} // namespace arm_compute
