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
#ifndef ARM_COMPUTE_TEST_TESTFILTER
#define ARM_COMPUTE_TEST_TESTFILTER

#include "DatasetModes.h"

#include <regex>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
struct TestInfo;

/** Test filter class.
 *
 * Stores information about which test cases are selected for execution. Based
 * on test name and test id.
 */
class TestFilter final
{
public:
    /** Default constructor. All tests selected. */
    TestFilter() = default;

    /** Constructor.
     *
     * The id_filter string has be a comma separated list of test ids. ... can
     * be used to include a range of tests. For instance, "..., 15" means all
     * test up to and including 15, "3, 6, ..., 10" means tests 3 and 6 to 10,
     * and "15, ..." means test 15 and all following.
     *
     * @param[in] mode        Dataset mode.
     * @param[in] name_filter Regular expression to filter tests by name. Only matching tests will be executed.
     * @param[in] id_filter   String to match selected test ids. Only matching tests will be executed.
     */
    TestFilter(DatasetMode mode, const std::string &name_filter, const std::string &id_filter);

    /** Check if a test case is selected to be executed.
     *
     * @param[in] info Test case info.
     *
     * @return True if the test case is selected to be executed.
     */
    bool is_selected(const TestInfo &info) const;

private:
    using Ranges = std::vector<std::pair<int, int>>;
    Ranges parse_id_filter(const std::string &id_filter) const;

    DatasetMode _dataset_mode{ DatasetMode::ALL };
    std::regex  _name_filter{ ".*" };
    Ranges      _id_filter{};
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_TESTFILTER */
