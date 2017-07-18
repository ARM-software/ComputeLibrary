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
#ifndef ARM_COMPUTE_TEST_FRAMEWORK_ASSERTS
#define ARM_COMPUTE_TEST_FRAMEWORK_ASSERTS

#include "Exceptions.h"
#include "Framework.h"

#include <sstream>
#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace framework
{
namespace detail
{
// Cast char values to int so that their numeric value are printed.
inline int make_printable(int8_t value)
{
    return value;
}

inline unsigned int make_printable(uint8_t value)
{
    return value;
}

// Everything else can be printed as its own type.
template <typename T>
inline T &&make_printable(T &&value)
{
    return value;
}

#define ARM_COMPUTE_TEST_INFO(INFO)                                               \
    {                                                                             \
        std::stringstream info;                                                   \
        info << INFO;                                                             \
        arm_compute::test::framework::Framework::get().add_test_info(info.str()); \
    }

#define ARM_COMPUTE_TEST_COMP_FACTORY(SEVERITY, SEVERITY_NAME, COMP, COMP_NAME, ERROR_CALL)                            \
    template <typename T, typename U>                                                                                  \
    void ARM_COMPUTE_##SEVERITY##_##COMP_NAME##_IMPL(T &&x, U &&y, const std::string &x_str, const std::string &y_str) \
    {                                                                                                                  \
        if(!(x COMP y))                                                                                                \
        {                                                                                                              \
            std::stringstream msg;                                                                                     \
            msg << #SEVERITY_NAME " '" << x_str << " " #COMP " " << y_str << "' failed. ["                             \
                << std::boolalpha << arm_compute::test::framework::detail::make_printable(x)                           \
                << " " #COMP " "                                                                                       \
                << std::boolalpha << arm_compute::test::framework::detail::make_printable(y)                           \
                << "]\n";                                                                                              \
            arm_compute::test::framework::Framework::get().print_test_info(msg);                                       \
            ERROR_CALL                                                                                                 \
        }                                                                                                              \
        arm_compute::test::framework::Framework::get().clear_test_info();                                              \
    }

ARM_COMPUTE_TEST_COMP_FACTORY(EXPECT, Expectation, ==, EQUAL, arm_compute::test::framework::Framework::get().log_failed_expectation(msg.str());)
ARM_COMPUTE_TEST_COMP_FACTORY(EXPECT, Expectation, !=, NOT_EQUAL, arm_compute::test::framework::Framework::get().log_failed_expectation(msg.str());)
ARM_COMPUTE_TEST_COMP_FACTORY(ASSERT, Assertion, ==, EQUAL, throw arm_compute::test::framework::TestError(msg.str());)
ARM_COMPUTE_TEST_COMP_FACTORY(ASSERT, Assertion, !=, NOT_EQUAL, throw arm_compute::test::framework::TestError(msg.str());)

#define ARM_COMPUTE_ASSERT_NOT_EQUAL(X, Y) \
    arm_compute::test::framework::detail::ARM_COMPUTE_ASSERT_NOT_EQUAL_IMPL(X, Y, #X, #Y)

#define ARM_COMPUTE_ASSERT_EQUAL(X, Y) \
    arm_compute::test::framework::detail::ARM_COMPUTE_ASSERT_EQUAL_IMPL(X, Y, #X, #Y)

#define ARM_COMPUTE_EXPECT_EQUAL(X, Y) \
    arm_compute::test::framework::detail::ARM_COMPUTE_EXPECT_EQUAL_IMPL(X, Y, #X, #Y)

#define ARM_COMPUTE_EXPECT_NOT_EQUAL(X, Y) \
    arm_compute::test::framework::detail::ARM_COMPUTE_EXPECT_NOT_EQUAL_IMPL(X, Y, #X, #Y)

#define ARM_COMPUTE_ASSERT(X)                                                    \
    do                                                                           \
    {                                                                            \
        const auto &x = X;                                                       \
        if(!x)                                                                   \
        {                                                                        \
            std::stringstream msg;                                               \
            msg << "Assertion '" #X "' failed.\n";                               \
            arm_compute::test::framework::Framework::get().print_test_info(msg); \
            throw arm_compute::test::framework::TestError(msg.str());            \
        }                                                                        \
        arm_compute::test::framework::Framework::get().clear_test_info();        \
    } while(false)

#define ARM_COMPUTE_EXPECT(X)                                                                 \
    do                                                                                        \
    {                                                                                         \
        const auto &x = X;                                                                    \
        if(!x)                                                                                \
        {                                                                                     \
            std::stringstream msg;                                                            \
            msg << "Expectation '" #X "' failed.\n";                                          \
            arm_compute::test::framework::Framework::get().print_test_info(msg);              \
            arm_compute::test::framework::Framework::get().log_failed_expectation(msg.str()); \
        }                                                                                     \
        arm_compute::test::framework::Framework::get().clear_test_info();                     \
    } while(false)
} // namespace detail
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FRAMEWORK_ASSERTS */
