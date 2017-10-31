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
#ifndef ARM_COMPUTE_TEST_FRAMEWORK
#define ARM_COMPUTE_TEST_FRAMEWORK

#include "DatasetModes.h"
#include "Exceptions.h"
#include "Profiler.h"
#include "TestCase.h"
#include "TestCaseFactory.h"
#include "TestFilter.h"
#include "TestResult.h"
#include "Utils.h"
#include "instruments/Instruments.h"
#include "printers/Printer.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Information about a test case.
 *
 * A test can be identified either via its id or via its name. Additionally
 * each test is tagged with the data set mode in which it will be used and
 * its status.
 *
 * @note The mapping between test id and test name is not guaranteed to be
 * stable. It is subject to change as new test are added.
 */
struct TestInfo
{
    int                     id;
    std::string             name;
    DatasetMode             mode;
    TestCaseFactory::Status status;
};

inline bool operator<(const TestInfo &lhs, const TestInfo &rhs)
{
    return lhs.id < rhs.id;
}

/** Main framework class.
 *
 * Keeps track of the global state, owns all test cases and collects results.
 */
class Framework final
{
public:
    /** Access to the singleton.
     *
     * @return Unique instance of the framework class.
     */
    static Framework &get();

    /** Supported instrument types for benchmarking.
     *
     * @return Set of all available instrument types.
     */
    std::set<InstrumentsDescription> available_instruments() const;

    /** Init the framework.
     *
     * @see TestFilter::TestFilter for the format of the string to filter ids.
     *
     * @param[in] instruments    Instrument types that will be used for benchmarking.
     * @param[in] num_iterations Number of iterations per test.
     * @param[in] mode           Dataset mode.
     * @param[in] name_filter    Regular expression to filter tests by name. Only matching tests will be executed.
     * @param[in] id_filter      String to match selected test ids. Only matching tests will be executed.
     * @param[in] log_level      Verbosity of the output.
     */
    void init(const std::vector<framework::InstrumentsDescription> &instruments, int num_iterations, DatasetMode mode, const std::string &name_filter, const std::string &id_filter, LogLevel log_level);

    /** Add a new test suite.
     *
     * @warning Cannot be used at execution time. It can only be used for
     * registering test cases.
     *
     * @param[in] name Name of the added test suite.
     *
     * @return Name of the current test suite.
     */
    void push_suite(std::string name);

    /** Remove innermost test suite.
     *
     * @warning Cannot be used at execution time. It can only be used for
     * registering test cases.
     */
    void pop_suite();

    /** Add a test case to the framework.
     *
     * @param[in] test_name Name of the new test case.
     * @param[in] mode      Mode in which to include the test.
     * @param[in] status    Status of the test case.
     */
    template <typename T>
    void add_test_case(std::string test_name, DatasetMode mode, TestCaseFactory::Status status);

    /** Add a data test case to the framework.
     *
     * @param[in] test_name   Name of the new test case.
     * @param[in] mode        Mode in which to include the test.
     * @param[in] status      Status of the test case.
     * @param[in] description Description of @p data.
     * @param[in] data        Data that will be used as input to the test.
     */
    template <typename T, typename D>
    void add_data_test_case(std::string test_name, DatasetMode mode, TestCaseFactory::Status status, std::string description, D &&data);

    /** Add info string for the next expectation/assertion.
     *
     * @param[in] info Info string.
     */
    void add_test_info(std::string info);

    /** Clear the collected test info. */
    void clear_test_info();

    /** Check if any info has been registered.
     *
     * @return True if there is test info.
     */
    bool has_test_info() const;

    /** Print test info.
     *
     * @param[out] os Output stream.
     */
    void print_test_info(std::ostream &os) const;

    /** Tell the framework that execution of a test starts.
     *
     * @param[in] info Test info.
     */
    void log_test_start(const TestInfo &info);

    /** Tell the framework that a test case is skipped.
     *
     * @param[in] info Test info.
     */
    void log_test_skipped(const TestInfo &info);

    /** Tell the framework that a test case finished.
     *
     * @param[in] info Test info.
     */
    void log_test_end(const TestInfo &info);

    /** Tell the framework that the currently running test case failed a non-fatal expectation.
     *
     * @param[in] error Description of the error.
     */
    void log_failed_expectation(const TestError &error);

    /** Print the debug information that has already been logged
     *
     * @param[in] info Description of the log info.
     */
    void log_info(const std::string &info);

    /** Number of iterations per test case.
     *
     * @return Number of iterations per test case.
     */
    int num_iterations() const;

    /** Set number of iterations per test case.
     *
     * @param[in] num_iterations Number of iterations per test case.
     */
    void set_num_iterations(int num_iterations);

    /** Should errors be caught or thrown by the framework.
     *
     * @return True if errors are thrown.
     */
    bool throw_errors() const;

    /** Set whether errors are caught or thrown by the framework.
     *
     * @param[in] throw_errors True if errors should be thrown.
     */
    void set_throw_errors(bool throw_errors);

    /** Indicates if test execution is stopped after the first failed test.
     *
     * @return True if the execution is going to be aborted after the first failed test.
     */
    bool stop_on_error() const;

    /** Set whether to abort execution after the first failed test.
     *
     * @param[in] stop_on_error True if execution is going to be aborted after first failed test.
     */
    void set_stop_on_error(bool stop_on_error);

    /** Indicates if a test should be marked as failed when its assets are missing.
     *
     * @return True if a test should be marked as failed when its assets are missing.
     */
    bool error_on_missing_assets() const;

    /** Set whether a test should be considered as failed if its assets cannot be found.
     *
     * @param[in] error_on_missing_assets True if a test should be marked as failed when its assets are missing.
     */
    void set_error_on_missing_assets(bool error_on_missing_assets);

    /** Run all enabled test cases.
     *
     * @return True if all test cases executed successful.
     */
    bool run();

    /** Set the result for an executed test case.
     *
     * @param[in] info   Test info.
     * @param[in] result Execution result.
     */
    void set_test_result(TestInfo info, TestResult result);

    /** Use the specified printer to output test results from the last run.
     *
     * This method can be used if the test results need to be obtained using a
     * different printer than the one managed by the framework.
     *
     * @param[in] printer Printer used to output results.
     */
    void print_test_results(Printer &printer) const;

    /** Factory method to obtain a configured profiler.
     *
     * The profiler enables all instruments that have been passed to the @ref
     * init method.
     *
     * @return Configured profiler to collect benchmark results.
     */
    Profiler get_profiler() const;

    /** Set the printer used for the output of test results.
     *
     * @param[in] printer Pointer to a printer.
     */
    void add_printer(Printer *printer);

    /** List of @ref TestInfo's.
     *
     * @return Vector with all test ids.
     */
    std::vector<TestInfo> test_infos() const;

    /** Get the current logging level
     *
     * @return The current logging level.
     */
    LogLevel log_level() const;

private:
    Framework();
    ~Framework() = default;

    Framework(const Framework &) = delete;
    Framework &operator=(const Framework &) = delete;

    void run_test(const TestInfo &info, TestCaseFactory &test_factory);
    std::map<TestResult::Status, int> count_test_results() const;

    /** Returns the current test suite name.
     *
     * @warning Cannot be used at execution time to get the test suite of the
     * currently executed test case. It can only be used for registering test
     * cases.
     *
     * @return Name of the current test suite.
     */
    std::string current_suite_name() const;

    /* Perform func on all printers */
    template <typename F>
    void func_on_all_printers(F &&func);

    std::vector<std::string>                      _test_suite_name{};
    std::vector<std::unique_ptr<TestCaseFactory>> _test_factories{};
    std::map<TestInfo, TestResult> _test_results{};
    int                    _num_iterations{ 1 };
    bool                   _throw_errors{ false };
    bool                   _stop_on_error{ false };
    bool                   _error_on_missing_assets{ false };
    std::vector<Printer *> _printers{};

    using create_function = std::unique_ptr<Instrument>();
    std::map<InstrumentsDescription, create_function *> _available_instruments{};

    std::set<framework::InstrumentsDescription> _instruments{ std::pair<InstrumentType, ScaleFactor>(InstrumentType::NONE, ScaleFactor::NONE) };
    TestFilter                                  _test_filter{};
    LogLevel                                    _log_level{ LogLevel::ALL };
    const TestInfo                             *_current_test_info{ nullptr };
    TestResult                                 *_current_test_result{ nullptr };
    std::vector<std::string>                    _test_info{};
};

template <typename T>
inline void Framework::add_test_case(std::string test_name, DatasetMode mode, TestCaseFactory::Status status)
{
    _test_factories.emplace_back(support::cpp14::make_unique<SimpleTestCaseFactory<T>>(current_suite_name(), std::move(test_name), mode, status));
}

template <typename T, typename D>
inline void Framework::add_data_test_case(std::string test_name, DatasetMode mode, TestCaseFactory::Status status, std::string description, D &&data)
{
    // WORKAROUND for GCC 4.9
    // The function should get *it which is tuple but that seems to trigger a
    // bug in the compiler.
    auto tmp = std::unique_ptr<DataTestCaseFactory<T, decltype(*std::declval<D>())>>(new DataTestCaseFactory<T, decltype(*std::declval<D>())>(current_suite_name(), std::move(test_name), mode, status,
                                                                                     std::move(description), *data));
    _test_factories.emplace_back(std::move(tmp));
}
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FRAMEWORK */
