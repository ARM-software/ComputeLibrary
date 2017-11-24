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
#include "JSONPrinter.h"

#include "../Framework.h"
#include "../instruments/Measurement.h"

#include <algorithm>

namespace arm_compute
{
namespace test
{
namespace framework
{
void JSONPrinter::print_separator(bool &flag)
{
    if(flag)
    {
        flag = false;
    }
    else
    {
        *_stream << ",";
    }
}

template <typename T>
void JSONPrinter::print_strings(T &&first, T &&last)
{
    bool              first_entry = true;
    std::stringstream log;

    while(first != last)
    {
        print_separator(first_entry);

        *_stream << R"(")";

        log.str(*first);

        for(std::string line; !std::getline(log, line).fail();)
        {
            *_stream << line << "; ";
        }

        *_stream << R"(")";

        ++first;
    }
}

void JSONPrinter::print_entry(const std::string &name, const std::string &value)
{
    print_separator(_first_entry);

    *_stream << R"(")" << name << R"(" : ")" << value << R"(")";
}

void JSONPrinter::print_global_header()
{
    *_stream << "{";
}

void JSONPrinter::print_global_footer()
{
    *_stream << "}\n";
}

void JSONPrinter::print_run_header()
{
    print_separator(_first_entry);

    *_stream << R"("tests" : {)";
}

void JSONPrinter::print_run_footer()
{
    *_stream << "}";
}

void JSONPrinter::print_test_header(const TestInfo &info)
{
    print_separator(_first_test);

    _first_test_entry = true;
    *_stream << R"(")" << info.name << R"(" : {)";
}

void JSONPrinter::print_test_footer()
{
    *_stream << "}";
}

void JSONPrinter::print_list_tests(const std::vector<TestInfo> &infos)
{
    *_stream << R"(, "list_tests" : {)";
    bool first = true;
    for(auto info : infos)
    {
        if(!first)
        {
            *_stream << ",";
        }
        *_stream << R"(")" << info.id << R"(" : { "name": ")" << info.name << R"(", "mode": ")" << info.mode << R"(", "status" : ")" << info.status << R"(" })";
        first = false;
    }
    *_stream << "}";
}
void JSONPrinter::print_errors_header()
{
    _errors.clear();
    _expected_errors.clear();
    _infos.clear();
}

void JSONPrinter::print_errors_footer()
{
    print_separator(_first_test_entry);

    *_stream << R"("errors" : [)";
    print_strings(_errors.begin(), _errors.end());
    *_stream << "]";

    *_stream << R"(, "expected_errors" : [)";
    print_strings(_expected_errors.begin(), _expected_errors.end());
    *_stream << "]";

    *_stream << R"(, "infos" : [)";
    print_strings(_infos.begin(), _infos.end());
    *_stream << "]";
}

void JSONPrinter::print_error(const std::exception &error, bool expected)
{
    if(expected)
    {
        _expected_errors.emplace_back(error.what());
    }
    else
    {
        _errors.emplace_back(error.what());
    }
}

void JSONPrinter::print_info(const std::string &info)
{
    _infos.push_back(info);
}

void JSONPrinter::print_measurements(const Profiler::MeasurementsMap &measurements)
{
    print_separator(_first_test_entry);

    *_stream << R"("measurements" : {)";

    for(auto i_it = measurements.cbegin(), i_end = measurements.cend(); i_it != i_end;)
    {
        *_stream << R"(")" << i_it->first << R"(" : {)";

        auto add_measurements = [](Measurement::Value a, const Measurement & b)
        {
            return a + b.value();
        };

        auto cmp_measurements = [](const Measurement & a, const Measurement & b)
        {
            return a.value() < b.value();
        };

        int        num_values    = i_it->second.size();
        const auto minmax_values = std::minmax_element(i_it->second.begin(), i_it->second.end(), cmp_measurements);

        Measurement::Value sum_values = std::accumulate(i_it->second.cbegin(), i_it->second.cend(), Measurement::Value(minmax_values.first->value().is_floating_point), add_measurements);
        if(num_values > 2)
        {
            sum_values -= minmax_values.first->value() + minmax_values.second->value();
            num_values -= 2;
        }

        auto measurement_to_string = [](const Measurement & measurement)
        {
            if(measurement.raw_data().size() == 1)
            {
                return measurement.raw_data().front();
            }
            else
            {
                std::stringstream str;
                str << R"([")";
                str << join(measurement.raw_data().begin(), measurement.raw_data().end(), R"(",")");
                str << R"("])";
                return str.str();
            }
        };
        *_stream << R"("avg" : )" << (sum_values / num_values) << ",";
        if(num_values > 1)
        {
            *_stream << R"("min" : )" << minmax_values.first->value() << ",";
            *_stream << R"("max" : )" << minmax_values.second->value() << ",";
        }
        *_stream << R"("raw" : [)" << join(i_it->second.begin(), i_it->second.end(), ",", measurement_to_string) << "],";
        *_stream << R"("unit" : ")" << minmax_values.first->unit() << R"(")";
        *_stream << "}";

        if(++i_it != i_end)
        {
            *_stream << ",";
        }
    }

    *_stream << "}";
}
} // namespace framework
} // namespace test
} // namespace arm_compute
