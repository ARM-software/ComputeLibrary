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
#include "PrettyPrinter.h"

#include "../Framework.h"
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
    for(auto info : infos)
    {
        *_stream << "[" << info.id << ", " << info.mode << ", " << info.status << "] " << info.name << "\n";
    }
}
void PrettyPrinter::print_measurements(const Profiler::MeasurementsMap &measurements)
{
    for(const auto &instrument : measurements)
    {
        *_stream << begin_color("3") << "  " << instrument.first << ":";

        auto add_measurements = [](Measurement::Value a, const Measurement & b)
        {
            return a + b.value();
        };

        auto cmp_measurements = [](const Measurement & a, const Measurement & b)
        {
            return a.value() < b.value();
        };

        int                num_values    = instrument.second.size();
        const auto         minmax_values = std::minmax_element(instrument.second.begin(), instrument.second.end(), cmp_measurements);
        Measurement::Value sum_values    = std::accumulate(instrument.second.begin(), instrument.second.end(), Measurement::Value(minmax_values.first->value().is_floating_point), add_measurements);

        // Calculate the median value
        auto measurements = instrument.second;
        std::nth_element(measurements.begin(), measurements.begin() + (num_values / 2), measurements.end(), cmp_measurements);
        const auto median_value = measurements[num_values / 2];

        // Calculate the relative standard deviation
        auto                            mean_value = sum_values / num_values;
        std::vector<Measurement::Value> diff(measurements.size(), minmax_values.first->value().is_floating_point);
        std::transform(measurements.begin(), measurements.end(), diff.begin(), [mean_value](const Measurement & x)
        {
            return x.value() - mean_value;
        });
        auto sq_sum   = std::inner_product(diff.begin(), diff.end(), diff.begin(), Measurement::Value(minmax_values.first->value().is_floating_point));
        auto variance = sq_sum / measurements.size();
        auto rsd      = Measurement::Value::relative_standard_deviation(variance, mean_value);

        if(num_values > 2)
        {
            sum_values -= minmax_values.first->value() + minmax_values.second->value();
            num_values -= 2;
        }

        *_stream << "    ";
        *_stream << "MEDIAN=" << median_value.value() << " " << median_value.unit() << ", ";
        *_stream << "AVG=" << (sum_values / num_values) << " " << minmax_values.second->unit() << ", ";
        *_stream << "STDDEV=" << arithmetic_to_string(rsd, 2) << " %, ";
        if(num_values > 1)
        {
            *_stream << "MIN=" << *minmax_values.first << ", ";
            *_stream << "MAX=" << *minmax_values.second;
        }
        *_stream << end_color() << "\n";
    }
}
} // namespace framework
} // namespace test
} // namespace arm_compute
