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
#ifndef ARM_COMPUTE_TEST_FRAMEWORK_REGISTRARS
#define ARM_COMPUTE_TEST_FRAMEWORK_REGISTRARS

#include "DatasetModes.h"
#include "Framework.h"

#include <string>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace framework
{
namespace detail
{
/** Helper class to statically register a test case. */
template <typename T>
class TestCaseRegistrar final
{
public:
    /** Add a new test case with the given name to the framework.
     *
     * @param[in] test_name Name of the test case.
     * @param[in] mode      Mode in which the test should be activated.
     * @param[in] status    Status of the test case.
     */
    TestCaseRegistrar(std::string test_name, DatasetMode mode, TestCaseFactory::Status status);

    /** Add a new data test case with the given name to the framework.
     *
     * @param[in] test_name Name of the test case.
     * @param[in] mode      Mode in which the test should be activated.
     * @param[in] status    Status of the test case.
     * @param[in] dataset   Dataset used as input for the test case.
     */
    template <typename D>
    TestCaseRegistrar(std::string test_name, DatasetMode mode, TestCaseFactory::Status status, D &&dataset);
};

/** Helper class to statically begin and end a test suite. */
class TestSuiteRegistrar final
{
public:
    /** Remove the last added test suite from the framework. */
    TestSuiteRegistrar();

    /** Add a new test suite with the given name to the framework.
     *
     * @param[in] name Name of the test suite.
     */
    TestSuiteRegistrar(std::string name);
};

template <typename T>
inline TestCaseRegistrar<T>::TestCaseRegistrar(std::string test_name, DatasetMode mode, TestCaseFactory::Status status)
{
    Framework::get().add_test_case<T>(std::move(test_name), mode, status);
}

template <typename T>
template <typename D>
inline TestCaseRegistrar<T>::TestCaseRegistrar(std::string test_name, DatasetMode mode, TestCaseFactory::Status status, D &&dataset)
{
    auto it = dataset.begin();

    for(int i = 0; i < dataset.size(); ++i, ++it)
    {
        // WORKAROUND for GCC 4.9
        // The last argument should be *it to pass just the data and not the
        // iterator.
        Framework::get().add_data_test_case<T>(test_name, mode, status, it.description(), it);
    }
}

inline TestSuiteRegistrar::TestSuiteRegistrar()
{
    Framework::get().pop_suite();
}

inline TestSuiteRegistrar::TestSuiteRegistrar(std::string name)
{
    Framework::get().push_suite(std::move(name));
}
} // namespace detail
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FRAMEWORK_REGISTRARS */
