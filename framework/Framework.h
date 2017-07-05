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

#include "TestCase.h"
#include "TestCaseFactory.h"
#include "TestResult.h"
#include "Utils.h"

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
/** Main framework class.
 *
 * Keeps track of the global state, owns all test cases and collects results.
 */
class Framework final
{
public:
    /** Type of a test identifier.
     *
     * A test can be identified either via its id or via its name.
     *
     * @note The mapping between test id and test name is not guaranteed to be
     * stable. It is subject to change as new test are added.
     */
    using TestId = std::pair<int, std::string>;

    /** Access to the singleton.
     *
     * @return Unique instance of the framework class.
     */
    static Framework &get();

    /** Init the framework.
     *
     * @param[in] num_iterations Number of iterations per test.
     * @param[in] name_filter    Regular expression to filter tests by name. Only matching tests will be executed.
     * @param[in] id_filter      Regular expression to filter tests by id. Only matching tests will be executed.
     */
    void init(int num_iterations, const std::string &name_filter, const std::string &id_filter);

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
     */
    template <typename T>
    void add_test_case(std::string test_name);

    /** Add a data test case to the framework.
     *
     * @param[in] test_name   Name of the new test case.
     * @param[in] description Description of @p data.
     * @param[in] data        Data that will be used as input to the test.
     */
    template <typename T, typename D>
    void add_data_test_case(std::string test_name, std::string description, D &&data);

    /** Tell the framework that execution of a test starts.
     *
     * @param[in] test_name Name of the started test case.
     */
    void log_test_start(const std::string &test_name);

    /** Tell the framework that a test case is skipped.
     *
     * @param[in] test_name Name of the skipped test case.
     */
    void log_test_skipped(const std::string &test_name);

    /** Tell the framework that a test case finished.
     *
     * @param[in] test_name Name of the finished test case.
     */
    void log_test_end(const std::string &test_name);

    /** Tell the framework that the currently running test case failed a non-fatal expectation.
     *
     * @param[in] msg Description of the failure.
     */
    void log_failed_expectation(const std::string &msg);

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

    /** Check if a test case would be executed.
     *
     * @param[in] id Id of the test case.
     *
     * @return True if the test case would be executed.
     */
    bool is_enabled(const TestId &id) const;

    /** Run all enabled test cases.
     *
     * @return True if all test cases executed successful.
     */
    bool run();

    /** Set the result for an executed test case.
     *
     * @param[in] test_case_name Name of the executed test case.
     * @param[in] result         Execution result.
     */
    void set_test_result(std::string test_case_name, TestResult result);

    /** List of @ref TestId's.
     *
     * @return Vector with all test ids.
     */
    std::vector<Framework::TestId> test_ids() const;

private:
    Framework()  = default;
    ~Framework() = default;

    Framework(const Framework &) = delete;
    Framework &operator=(const Framework &) = delete;

    void run_test(TestCaseFactory &test_factory);
    std::tuple<int, int, int> count_test_results() const;

    /** Returns the current test suite name.
     *
     * @warning Cannot be used at execution time to get the test suite of the
     * currently executed test case. It can only be used for registering test
     * cases.
     *
     * @return Name of the current test suite.
     */
    std::string current_suite_name() const;

    std::vector<std::string>                      _test_suite_name{};
    std::vector<std::unique_ptr<TestCaseFactory>> _test_factories{};
    std::map<std::string, TestResult> _test_results{};
    std::chrono::seconds _runtime{ 0 };
    int                  _num_iterations{ 1 };
    bool                 _throw_errors{ false };

    std::regex _test_name_filter{ ".*" };
    std::regex _test_id_filter{ ".*" };
};

template <typename T>
inline void Framework::add_test_case(std::string test_name)
{
    _test_factories.emplace_back(support::cpp14::make_unique<SimpleTestCaseFactory<T>>(current_suite_name(), std::move(test_name)));
}

template <typename T, typename D>
inline void Framework::add_data_test_case(std::string test_name, std::string description, D &&data)
{
    // WORKAROUND for GCC 4.9
    // The function should get *it which is tuple but that seems to trigger a
    // bug in the compiler.
    auto tmp = std::unique_ptr<DataTestCaseFactory<T, decltype(*std::declval<D>())>>(new DataTestCaseFactory<T, decltype(*std::declval<D>())>(current_suite_name(), std::move(test_name),
                                                                                     std::move(description), *data));
    _test_factories.emplace_back(std::move(tmp));
}
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FRAMEWORK */
