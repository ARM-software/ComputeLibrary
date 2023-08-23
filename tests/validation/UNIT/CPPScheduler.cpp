/*
 * Copyright (c) 2023 Arm Limited.
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
#include "arm_compute/runtime/CPP/CPPScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

#include <stdexcept>

using namespace arm_compute;
using namespace arm_compute::test;

namespace
{
class TestException: public std::exception
{
public:
    const char* what() const noexcept override
    {
        return "Expected test exception";
    }
};

class TestKernel: public ICPPKernel
{
public:
    TestKernel()
    {
        Window window;
        window.set(0, Window::Dimension(0, 2));
        configure(window);
    }

    const char* name() const override
    {
        return "TestKernel";
    }

    void run(const Window &, const ThreadInfo &) override
    {
        throw TestException();
    }

};
}

TEST_SUITE(UNIT)
TEST_SUITE(CPPScheduler)

#if !defined(BARE_METAL)
TEST_CASE(RethrowException, framework::DatasetMode::ALL)
{
    CPPScheduler scheduler;
    CPPScheduler::Hints hints(0);
    TestKernel kernel;

    scheduler.set_num_threads(2);
    try
    {
        scheduler.schedule(&kernel, hints);
    }
    catch(const TestException&)
    {
        return;
    }
    ARM_COMPUTE_EXPECT_FAIL("Expected exception not caught", framework::LogLevel::ERRORS);
}
#endif // !defined(BARE_METAL)

TEST_SUITE_END()
TEST_SUITE_END()
