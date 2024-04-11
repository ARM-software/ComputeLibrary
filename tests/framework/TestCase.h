/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_TESTCASE
#define ARM_COMPUTE_TEST_TESTCASE

#include <string>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Abstract test case class.
 *
 * All test cases have to inherit from this class.
 */
class TestCase
{
public:
    /** Setup the test */
    virtual void do_setup() {};
    /** Run the test */
    virtual void do_run() {};
    /** Sync the test */
    virtual void do_sync() {};
    /** Teardown the test */
    virtual void do_teardown() {};

    /** Default destructor. */
    virtual ~TestCase() = default;

protected:
    TestCase() = default;

    friend class TestCaseFactory;
};

/** Data test case class */
template <typename T>
class DataTestCase : public TestCase
{
protected:
    /** Construct a data test case.
     *
     * @param[in] data Test data.
     */
    explicit DataTestCase(T data)
        : _data{ std::move(data) }
    {
    }

    T _data;
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_TESTCASE */
