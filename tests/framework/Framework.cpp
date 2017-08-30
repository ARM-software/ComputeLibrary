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
#include "Framework.h"

#include "support/ToolchainSupport.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#endif /* ARM_COMPUTE_CL */

#include <chrono>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace framework
{
Framework::Framework()
{
    _available_instruments.emplace(InstrumentType::WALL_CLOCK_TIMER, Instrument::make_instrument<WallClockTimer>);
#ifdef PMU_ENABLED
    _available_instruments.emplace(InstrumentType::PMU, Instrument::make_instrument<PMUCounter>);
#endif /* PMU_ENABLED */
#ifdef MALI_ENABLED
    _available_instruments.emplace(InstrumentType::MALI, Instrument::make_instrument<MaliCounter>);
#endif /* MALI_ENABLED */
}

std::set<InstrumentType> Framework::available_instruments() const
{
    std::set<InstrumentType> types;

    for(const auto &instrument : _available_instruments)
    {
        types.emplace(instrument.first);
    }

    return types;
}

std::map<TestResult::Status, int> Framework::count_test_results() const
{
    std::map<TestResult::Status, int> counts;

    for(const auto &test : _test_results)
    {
        ++counts[test.second.status];
    }

    return counts;
}

Framework &Framework::get()
{
    static Framework instance;
    return instance;
}

void Framework::init(const std::vector<InstrumentType> &instruments, int num_iterations, DatasetMode mode, const std::string &name_filter, const std::string &id_filter, LogLevel log_level)
{
    _test_filter    = TestFilter(mode, name_filter, id_filter);
    _num_iterations = num_iterations;
    _log_level      = log_level;

    _instruments = std::set<InstrumentType>(instruments.begin(), instruments.end());
}

std::string Framework::current_suite_name() const
{
    return join(_test_suite_name.cbegin(), _test_suite_name.cend(), "/");
}

void Framework::push_suite(std::string name)
{
    _test_suite_name.emplace_back(std::move(name));
}

void Framework::pop_suite()
{
    _test_suite_name.pop_back();
}

void Framework::add_test_info(std::string info)
{
    _test_info.emplace_back(std::move(info));
}

void Framework::clear_test_info()
{
    _test_info.clear();
}

bool Framework::has_test_info() const
{
    return !_test_info.empty();
}

void Framework::print_test_info(std::ostream &os) const
{
    if(!_test_info.empty())
    {
        os << "CONTEXT:\n";

        for(const auto &str : _test_info)
        {
            os << "    " << str << "\n";
        }
    }
}

void Framework::log_test_start(const TestInfo &info)
{
    if(_printer != nullptr && _log_level >= LogLevel::TESTS)
    {
        _printer->print_test_header(info);
    }
}

void Framework::log_test_skipped(const TestInfo &info)
{
    static_cast<void>(info);
}

void Framework::log_test_end(const TestInfo &info)
{
    if(_printer != nullptr)
    {
        if(_log_level >= LogLevel::MEASUREMENTS)
        {
            _printer->print_measurements(_test_results.at(info).measurements);
        }

        if(_log_level >= LogLevel::TESTS)
        {
            _printer->print_test_footer();
        }
    }
}

void Framework::log_failed_expectation(const TestError &error)
{
    if(_log_level >= error.level() && _printer != nullptr)
    {
        constexpr bool expected_error = true;
        _printer->print_error(error, expected_error);
    }

    if(_current_test_result != nullptr)
    {
        _current_test_result->status = TestResult::Status::FAILED;
    }
}

void Framework::log_info(const std::string &info)
{
    if(_log_level >= LogLevel::DEBUG && _printer != nullptr)
    {
        _printer->print_info(info);
    }
}

int Framework::num_iterations() const
{
    return _num_iterations;
}

void Framework::set_num_iterations(int num_iterations)
{
    _num_iterations = num_iterations;
}

void Framework::set_throw_errors(bool throw_errors)
{
    _throw_errors = throw_errors;
}

bool Framework::throw_errors() const
{
    return _throw_errors;
}

void Framework::set_stop_on_error(bool stop_on_error)
{
    _stop_on_error = stop_on_error;
}

bool Framework::stop_on_error() const
{
    return _stop_on_error;
}

void Framework::run_test(const TestInfo &info, TestCaseFactory &test_factory)
{
    if(test_factory.status() == TestCaseFactory::Status::DISABLED)
    {
        log_test_skipped(info);
        set_test_result(info, TestResult(TestResult::Status::DISABLED));
        return;
    }

    log_test_start(info);

    Profiler   profiler = get_profiler();
    TestResult result(TestResult::Status::NOT_RUN);

    _current_test_result = &result;

    if(_log_level >= LogLevel::ERRORS && _printer != nullptr)
    {
        _printer->print_errors_header();
    }

    const bool is_expected_failure = test_factory.status() == TestCaseFactory::Status::EXPECTED_FAILURE;

    try
    {
        std::unique_ptr<TestCase> test_case = test_factory.make();

        try
        {
            test_case->do_setup();

            for(int i = 0; i < _num_iterations; ++i)
            {
                profiler.start();
                test_case->do_run();
#ifdef ARM_COMPUTE_CL
                if(opencl_is_available())
                {
                    CLScheduler::get().sync();
                }
#endif /* ARM_COMPUTE_CL */
                profiler.stop();
            }

            test_case->do_teardown();

            // Change status to success if no error has happend
            if(result.status == TestResult::Status::NOT_RUN)
            {
                result.status = TestResult::Status::SUCCESS;
            }
        }
        catch(const TestError &error)
        {
            if(_log_level >= error.level() && _printer != nullptr)
            {
                _printer->print_error(error, is_expected_failure);
            }

            result.status = TestResult::Status::FAILED;

            if(_throw_errors)
            {
                throw;
            }
        }
#ifdef ARM_COMPUTE_CL
        catch(const ::cl::Error &error)
        {
            if(_log_level >= LogLevel::ERRORS && _printer != nullptr)
            {
                std::stringstream stream;
                stream << "Error code: " << error.err();
                TestError test_error(error.what(), LogLevel::ERRORS, stream.str());
                _printer->print_error(test_error, is_expected_failure);
            }

            result.status = TestResult::Status::FAILED;

            if(_throw_errors)
            {
                throw;
            }
        }
#endif /* ARM_COMPUTE_CL */
        catch(const std::exception &error)
        {
            if(_log_level >= LogLevel::ERRORS && _printer != nullptr)
            {
                _printer->print_error(error, is_expected_failure);
            }

            result.status = TestResult::Status::CRASHED;

            if(_throw_errors)
            {
                throw;
            }
        }
        catch(...)
        {
            if(_log_level >= LogLevel::ERRORS && _printer != nullptr)
            {
                _printer->print_error(TestError("Received unknown exception"), is_expected_failure);
            }

            result.status = TestResult::Status::CRASHED;

            if(_throw_errors)
            {
                throw;
            }
        }
    }
    catch(const std::exception &error)
    {
        if(_log_level >= LogLevel::ERRORS && _printer != nullptr)
        {
            _printer->print_error(error, is_expected_failure);
        }

        result.status = TestResult::Status::CRASHED;

        if(_throw_errors)
        {
            throw;
        }
    }
    catch(...)
    {
        if(_log_level >= LogLevel::ERRORS && _printer != nullptr)
        {
            _printer->print_error(TestError("Received unknown exception"), is_expected_failure);
        }

        result.status = TestResult::Status::CRASHED;

        if(_throw_errors)
        {
            throw;
        }
    }

    if(_log_level >= LogLevel::ERRORS && _printer != nullptr)
    {
        _printer->print_errors_footer();
    }

    _current_test_result = nullptr;

    if(result.status == TestResult::Status::FAILED)
    {
        if(info.status == TestCaseFactory::Status::EXPECTED_FAILURE)
        {
            result.status = TestResult::Status::EXPECTED_FAILURE;
        }
    }

    if(result.status == TestResult::Status::FAILED || result.status == TestResult::Status::CRASHED)
    {
        if(_stop_on_error)
        {
            throw std::runtime_error("Abort on first error.");
        }
    }

    result.measurements = profiler.measurements();

    set_test_result(info, result);
    log_test_end(info);
}

bool Framework::run()
{
    // Clear old test results
    _test_results.clear();

    if(_printer != nullptr && _log_level >= LogLevel::TESTS)
    {
        _printer->print_run_header();
    }

    const std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    int id = 0;

    for(auto &test_factory : _test_factories)
    {
        const std::string test_case_name = test_factory->name();
        const TestInfo    test_info{ id, test_case_name, test_factory->mode(), test_factory->status() };

        if(_test_filter.is_selected(test_info))
        {
            run_test(test_info, *test_factory);
        }

        ++id;
    }

    const std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();

    if(_printer != nullptr && _log_level >= LogLevel::TESTS)
    {
        _printer->print_run_footer();
    }

    auto runtime = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::map<TestResult::Status, int> results = count_test_results();

    if(_log_level > LogLevel::NONE)
    {
        std::cout << "Executed " << _test_results.size() << " test(s) ("
                  << results[TestResult::Status::SUCCESS] << " passed, "
                  << results[TestResult::Status::EXPECTED_FAILURE] << " expected failures, "
                  << results[TestResult::Status::FAILED] << " failed, "
                  << results[TestResult::Status::CRASHED] << " crashed, "
                  << results[TestResult::Status::DISABLED] << " disabled) in " << runtime.count() << " second(s)\n";
    }

    int num_successful_tests = results[TestResult::Status::SUCCESS] + results[TestResult::Status::EXPECTED_FAILURE] + results[TestResult::Status::DISABLED];

    return (static_cast<unsigned int>(num_successful_tests) == _test_results.size());
}

void Framework::set_test_result(TestInfo info, TestResult result)
{
    _test_results.emplace(std::move(info), std::move(result));
}

void Framework::print_test_results(Printer &printer) const
{
    printer.print_run_header();

    for(const auto &test : _test_results)
    {
        printer.print_test_header(test.first);
        printer.print_measurements(test.second.measurements);
        printer.print_test_footer();
    }

    printer.print_run_footer();
}

Profiler Framework::get_profiler() const
{
    Profiler profiler;

    const bool all_instruments = std::any_of(
                                     _instruments.begin(),
                                     _instruments.end(),
                                     [](InstrumentType type) -> bool { return type == InstrumentType::ALL; });

    auto is_selected = [&](InstrumentType instrument) -> bool
    {
        return std::find_if(_instruments.begin(), _instruments.end(), [&](InstrumentType type) -> bool {
            const auto group = static_cast<InstrumentType>(static_cast<uint64_t>(type) & 0xFF00);
            return group == instrument;
        })
        != _instruments.end();
    };

    for(const auto &instrument : _available_instruments)
    {
        if(all_instruments || is_selected(instrument.first))
        {
            profiler.add(instrument.second());
        }
    }

    return profiler;
}

void Framework::set_printer(Printer *printer)
{
    _printer = printer;
}

std::vector<TestInfo> Framework::test_infos() const
{
    std::vector<TestInfo> ids;

    int id = 0;

    for(const auto &factory : _test_factories)
    {
        TestInfo test_info{ id, factory->name(), factory->mode(), factory->status() };

        if(_test_filter.is_selected(test_info))
        {
            ids.emplace_back(std::move(test_info));
        }

        ++id;
    }

    return ids;
}

LogLevel Framework::log_level() const
{
    return _log_level;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
