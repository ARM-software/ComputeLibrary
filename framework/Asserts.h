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
} // namespace detail
} // namespace framework
} // namespace arm_compute

template <typename T>
void ARM_COMPUTE_ASSERT_EQUAL(T &&x, T &&y)
{
    if(x != y)
    {
        std::stringstream msg;
        msg << "Assertion "
            << std::boolalpha << arm_compute::framework::detail::make_printable(x) << " == "
            << std::boolalpha << arm_compute::framework::detail::make_printable(y) << " failed.";
        throw arm_compute::test::framework::TestError(msg.str());
    }
}

template <typename T>
void ARM_COMPUTE_EXPECT_EQUAL(T &&x, T &&y)
{
    if(x != y)
    {
        std::stringstream msg;
        msg << "Expectation "
            << std::boolalpha << arm_compute::framework::detail::make_printable(x) << " == "
            << std::boolalpha << arm_compute::framework::detail::make_printable(y) << " failed.";
        arm_compute::test::framework::Framework::get().log_failed_expectation(msg.str());
    }
}

template <typename T>
void ARM_COMPUTE_ASSERT_NOT_EQUAL(T &&x, T &&y)
{
    if(x == y)
    {
        std::stringstream msg;
        msg << "Assertion "
            << std::boolalpha << arm_compute::framework::detail::make_printable(x) << " != "
            << std::boolalpha << arm_compute::framework::detail::make_printable(y) << " failed.";
        throw arm_compute::test::framework::TestError(msg.str());
    }
}

template <typename T>
void ARM_COMPUTE_EXPECT_NOT_EQUAL(T &&x, T &&y)
{
    if(x == y)
    {
        std::stringstream msg;
        msg << "Expectation "
            << std::boolalpha << arm_compute::framework::detail::make_printable(x) << " != "
            << std::boolalpha << arm_compute::framework::detail::make_printable(y) << " failed.";
        arm_compute::test::framework::Framework::get().log_failed_expectation(msg.str());
    }
}

#define ARM_COMPUTE_ASSERT(X, MSG)                                    \
    do                                                                \
    {                                                                 \
        const auto &x = X;                                            \
        if(!x)                                                        \
        {                                                             \
            std::stringstream msg;                                    \
            msg << "Failed assertion: " << #X << "\n"                 \
                << MSG;                                               \
            throw arm_compute::test::framework::TestError(msg.str()); \
        }                                                             \
    } while(false)

#define ARM_COMPUTE_EXPECT(X, MSG)                                                            \
    do                                                                                        \
    {                                                                                         \
        const auto &x = X;                                                                    \
        if(!x)                                                                                \
        {                                                                                     \
            std::stringstream msg;                                                            \
            msg << "Failed assertion: " << #X << "\n"                                         \
                << MSG;                                                                       \
            arm_compute::test::framework::Framework::get().log_failed_expectation(msg.str()); \
        }                                                                                     \
    } while(false)
#endif /* ARM_COMPUTE_TEST_FRAMEWORK_ASSERTS */
