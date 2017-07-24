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
#ifndef ARM_COMPUTE_TEST_TEST_CASE_FACTORY
#define ARM_COMPUTE_TEST_TEST_CASE_FACTORY

#include "DatasetModes.h"
#include "TestCase.h"
#include "support/ToolchainSupport.h"

#include <memory>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Abstract factory class to create test cases. */
class TestCaseFactory
{
public:
    /** Test case status.
     *
     * ACTIVE == Test is run and result is validated. Failure on failed validation.
     * EXPECTED_FAILURE == Test is run and result is validated. Failure on successful validation.
     * DISABLED == Test is not run.
     */
    enum class Status
    {
        ACTIVE,
        EXPECTED_FAILURE,
        DISABLED
    };

    /** Constructor.
     *
     * @param[in] suite_name  Name of the test suite to which the test case has been added.
     * @param[in] name        Name of the test case.
     * @param[in] mode        Datset mode of the test case.
     * @param[in] status      Status of the test case.
     * @param[in] description Description of data arguments.
     */
    TestCaseFactory(std::string suite_name, std::string name, DatasetMode mode, Status status, std::string description = "");

    /** Default destructor. */
    virtual ~TestCaseFactory() = default;

    /** Name of the test case.
     *
     * @return Name of the test case.
     */
    std::string name() const;

    /** Get the mode for which test case will be enabled.
     *
     * @return Dataset mode of the test case.
     */
    DatasetMode mode() const;

    /** Get the status of the test case.
     *
     * @return Status of the test case.
     */
    Status status() const;

    /** Factory function to create the test case
     *
     * @return Unique pointer to a newly created test case.
     */
    virtual std::unique_ptr<TestCase> make() const = 0;

private:
    const std::string _suite_name;
    const std::string _test_name;
    const std::string _data_description;
    const DatasetMode _mode{ DatasetMode::ALL };
    const Status      _status{ Status::ACTIVE };
};

/** Implementation of a test case factory to create non-data test cases. */
template <typename T>
class SimpleTestCaseFactory final : public TestCaseFactory
{
public:
    /** Default constructor. */
    using TestCaseFactory::TestCaseFactory;

    std::unique_ptr<TestCase> make() const override;
};

template <typename T, typename D>
class DataTestCaseFactory final : public TestCaseFactory
{
public:
    /** Constructor.
     *
     * @param[in] suite_name  Name of the test suite to which the test case has been added.
     * @param[in] test_name   Name of the test case.
     * @param[in] mode        Mode in which the test case is enabled.
     * @param[in] status      Status of the test case.
     * @param[in] description Description of data arguments.
     * @param[in] data        Input data for the test case.
     */
    DataTestCaseFactory(std::string suite_name, std::string test_name, DatasetMode mode, Status status, std::string description, const D &data);

    std::unique_ptr<TestCase> make() const override;

private:
    D _data;
};

inline TestCaseFactory::TestCaseFactory(std::string suite_name, std::string test_name, DatasetMode mode, Status status, std::string description)
    : _suite_name{ std::move(suite_name) }, _test_name{ std::move(test_name) }, _data_description{ std::move(description) }, _mode{ mode }, _status{ status }

{
}

inline std::string TestCaseFactory::name() const
{
    std::string name = _suite_name + "/" + _test_name;

    if(!_data_description.empty())
    {
        name += "@" + _data_description;
    }

    return name;
}

inline DatasetMode TestCaseFactory::mode() const
{
    return _mode;
}

inline TestCaseFactory::Status TestCaseFactory::status() const
{
    return _status;
}

inline ::std::ostream &operator<<(::std::ostream &stream, TestCaseFactory::Status status)
{
    switch(status)
    {
        case TestCaseFactory::Status::ACTIVE:
            stream << "ACTIVE";
            break;
        case TestCaseFactory::Status::EXPECTED_FAILURE:
            stream << "EXPECTED_FAILURE";
            break;
        case TestCaseFactory::Status::DISABLED:
            stream << "DISABLED";
            break;
        default:
            throw std::invalid_argument("Unsupported test case factory status");
    }

    return stream;
}

template <typename T>
inline std::unique_ptr<TestCase> SimpleTestCaseFactory<T>::make() const
{
    return support::cpp14::make_unique<T>();
}

template <typename T, typename D>
inline DataTestCaseFactory<T, D>::DataTestCaseFactory(std::string suite_name, std::string test_name, DatasetMode mode, Status status, std::string description, const D &data)
    : TestCaseFactory{ std::move(suite_name), std::move(test_name), mode, status, std::move(description) }, _data{ data }
{
}

template <typename T, typename D>
inline std::unique_ptr<TestCase> DataTestCaseFactory<T, D>::make() const
{
    return support::cpp14::make_unique<T>(_data);
}
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_TEST_CASE_FACTORY */
