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

#include "Exceptions.h"
#include "support/ToolchainSupport.h"

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
    _available_instruments.emplace(InstrumentType::PMU_CYCLE_COUNTER, Instrument::make_instrument<CycleCounter>);
    _available_instruments.emplace(InstrumentType::PMU_INSTRUCTION_COUNTER, Instrument::make_instrument<InstructionCounter>);
#endif /* PMU_ENABLED */
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

std::tuple<int, int, int> Framework::count_test_results() const
{
    int passed  = 0;
    int failed  = 0;
    int crashed = 0;

    for(const auto &test : _test_results)
    {
        switch(test.second.status)
        {
            case TestResult::Status::SUCCESS:
                ++passed;
                break;
            case TestResult::Status::FAILED:
                ++failed;
                break;
            case TestResult::Status::CRASHED:
                ++crashed;
                break;
            default:
                // Do nothing
                break;
        }
    }

    return std::make_tuple(passed, failed, crashed);
}

Framework &Framework::get()
{
    static Framework instance;
    return instance;
}

void Framework::init(const std::vector<InstrumentType> &instruments, int num_iterations, const std::string &name_filter, const std::string &id_filter)
{
    _test_name_filter = std::regex{ name_filter };
    _test_id_filter   = std::regex{ id_filter };
    _num_iterations   = num_iterations;

    _instruments = InstrumentType::NONE;

    for(const auto &instrument : instruments)
    {
        _instruments |= instrument;
    }
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

void Framework::log_test_start(const std::string &test_name)
{
    static_cast<void>(test_name);
}

void Framework::log_test_skipped(const std::string &test_name)
{
    static_cast<void>(test_name);
}

void Framework::log_test_end(const std::string &test_name)
{
    static_cast<void>(test_name);
}

void Framework::log_failed_expectation(const std::string &msg)
{
    std::cerr << "ERROR: " << msg << "\n";
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

bool Framework::is_enabled(const TestId &id) const
{
    return (std::regex_search(support::cpp11::to_string(id.first), _test_id_filter) && std::regex_search(id.second, _test_name_filter));
}

void Framework::run_test(TestCaseFactory &test_factory)
{
    const std::string test_case_name = test_factory.name();

    log_test_start(test_case_name);

    Profiler   profiler = get_profiler();
    TestResult result;

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
                profiler.stop();
            }

            test_case->do_teardown();

            result.status = TestResult::Status::SUCCESS;
        }
        catch(const TestError &error)
        {
            std::cerr << "FATAL ERROR: " << error.what() << "\n";
            result.status = TestResult::Status::FAILED;

            if(_throw_errors)
            {
                throw;
            }
        }
        catch(const std::exception &error)
        {
            std::cerr << "FATAL ERROR: Received unhandled error: '" << error.what() << "'\n";
            result.status = TestResult::Status::CRASHED;

            if(_throw_errors)
            {
                throw;
            }
        }
        catch(...)
        {
            std::cerr << "FATAL ERROR: Received unhandled exception\n";
            result.status = TestResult::Status::CRASHED;

            if(_throw_errors)
            {
                throw;
            }
        }
    }
    catch(const std::exception &error)
    {
        std::cerr << "FATAL ERROR: Received unhandled error during fixture creation: '" << error.what() << "'\n";

        if(_throw_errors)
        {
            throw;
        }
    }
    catch(...)
    {
        std::cerr << "FATAL ERROR: Received unhandled exception during fixture creation\n";

        if(_throw_errors)
        {
            throw;
        }
    }

    result.measurements = profiler.measurements();

    set_test_result(test_case_name, result);
    log_test_end(test_case_name);
}

bool Framework::run()
{
    // Clear old test results
    _test_results.clear();
    _runtime = std::chrono::seconds{ 0 };

    const auto start = std::chrono::high_resolution_clock::now();

    int id = 0;

    for(auto &test_factory : _test_factories)
    {
        const std::string test_case_name = test_factory->name();

        if(!is_enabled(TestId(id, test_case_name)))
        {
            log_test_skipped(test_case_name);
        }
        else
        {
            run_test(*test_factory);
        }

        ++id;
    }

    const auto end = std::chrono::high_resolution_clock::now();

    _runtime = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    int passed  = 0;
    int failed  = 0;
    int crashed = 0;

    std::tie(passed, failed, crashed) = count_test_results();

    std::cout << "Executed " << _test_results.size() << " test(s) (" << passed << " passed, " << failed << " failed, " << crashed << " crashed) in " << _runtime.count() << " second(s)\n";

    return (static_cast<unsigned int>(passed) == _test_results.size());
}

void Framework::set_test_result(std::string test_case_name, TestResult result)
{
    _test_results.emplace(std::move(test_case_name), std::move(result));
}

Profiler Framework::get_profiler() const
{
    Profiler profiler;

    for(const auto &instrument : _available_instruments)
    {
        if((instrument.first & _instruments) != InstrumentType::NONE)
        {
            profiler.add(instrument.second());
        }
    }

    return profiler;
}

std::vector<Framework::TestId> Framework::test_ids() const
{
    std::vector<TestId> ids;

    int id = 0;

    for(const auto &factory : _test_factories)
    {
        if(is_enabled(TestId(id, factory->name())))
        {
            ids.emplace_back(id, factory->name());
        }

        ++id;
    }

    return ids;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
