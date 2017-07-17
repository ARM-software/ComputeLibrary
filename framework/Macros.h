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
#ifndef ARM_COMPUTE_TEST_FRAMEWORK_MACROS
#define ARM_COMPUTE_TEST_FRAMEWORK_MACROS

#include "Exceptions.h"
#include "Framework.h"
#include "Registrars.h"
#include "TestCase.h"

#include <sstream>

//
// TEST SUITE MACROS
//
#define TEST_SUITE(SUITE_NAME)  \
    namespace SUITE_NAME##Suite \
    {                           \
    static arm_compute::test::framework::detail::TestSuiteRegistrar SUITE_NAME##Suite_reg{ #SUITE_NAME };

#define TEST_SUITE_END()                                                       \
    static arm_compute::test::framework::detail::TestSuiteRegistrar Suite_end; \
    }
//
// TEST SUITE MACROS END
//

//
// TEST CASE MACROS
//
#define TEST_CASE_CONSTRUCTOR(TEST_NAME) \
    TEST_NAME() = default;
#define DATA_TEST_CASE_CONSTRUCTOR(TEST_NAME, DATASET)                   \
    template <typename D>                                                \
    explicit TEST_NAME(D &&data) : DataTestCase{ std::forward<D>(data) } \
    {                                                                    \
    }
#define FIXTURE_SETUP(FIXTURE) \
    void do_setup() override   \
    {                          \
        FIXTURE::setup();      \
    }
#define FIXTURE_DATA_SETUP(FIXTURE)          \
    void do_setup() override                 \
    {                                        \
        apply(this, &FIXTURE::setup, _data); \
    }
#define FIXTURE_RUN(FIXTURE) \
    void do_run() override   \
    {                        \
        FIXTURE::run();      \
    }
#define FIXTURE_TEARDOWN(FIXTURE) \
    void do_teardown() override   \
    {                             \
        FIXTURE::teardown();      \
    }
#define TEST_REGISTRAR(TEST_NAME, MODE)                                                       \
    static arm_compute::test::framework::detail::TestCaseRegistrar<TEST_NAME> TEST_NAME##_reg \
    {                                                                                         \
        #TEST_NAME, MODE                                                                      \
    }
#define DATA_TEST_REGISTRAR(TEST_NAME, MODE, DATASET)                                         \
    static arm_compute::test::framework::detail::TestCaseRegistrar<TEST_NAME> TEST_NAME##_reg \
    {                                                                                         \
        #TEST_NAME, MODE, DATASET                                                             \
    }

#define TEST_CASE(TEST_NAME, MODE)                                  \
    class TEST_NAME : public arm_compute::test::framework::TestCase \
    {                                                               \
    public:                                                     \
        TEST_CASE_CONSTRUCTOR(TEST_NAME)                            \
        void do_run() override;                                     \
    };                                                              \
    TEST_REGISTRAR(TEST_NAME, MODE);                                \
    void TEST_NAME::do_run()

#define DATA_TEST_CASE(TEST_NAME, MODE, DATASET, ...)                                            \
    class TEST_NAME : public arm_compute::test::framework::DataTestCase<decltype(DATASET)::type> \
    {                                                                                            \
    public:                                                                                  \
        DATA_TEST_CASE_CONSTRUCTOR(TEST_NAME, DATASET)                                           \
        void do_run() override                                                                   \
        {                                                                                        \
            arm_compute::test::framework::apply(this, &TEST_NAME::run, _data);                   \
        }                                                                                        \
        void run(__VA_ARGS__);                                                                   \
    };                                                                                           \
    DATA_TEST_REGISTRAR(TEST_NAME, MODE, DATASET);                                               \
    void TEST_NAME::run(__VA_ARGS__)

#define FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE)                                 \
    class TEST_NAME : public arm_compute::test::framework::TestCase, public FIXTURE \
    {                                                                               \
    public:                                                                     \
        TEST_CASE_CONSTRUCTOR(TEST_NAME)                                            \
        FIXTURE_SETUP(FIXTURE)                                                      \
        void do_run() override;                                                     \
        FIXTURE_TEARDOWN(FIXTURE)                                                   \
    };                                                                              \
    TEST_REGISTRAR(TEST_NAME, MODE);                                                \
    void TEST_NAME::do_run()

#define FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET)                                                \
    class TEST_NAME : public arm_compute::test::framework::DataTestCase<decltype(DATASET)::type>, public FIXTURE \
    {                                                                                                            \
    public:                                                                                                  \
        DATA_TEST_CASE_CONSTRUCTOR(TEST_NAME, DATASET)                                                           \
        FIXTURE_DATA_SETUP(FIXTURE)                                                                              \
        void do_run() override;                                                                                  \
        FIXTURE_TEARDOWN(FIXTURE)                                                                                \
    };                                                                                                           \
    DATA_TEST_REGISTRAR(TEST_NAME, MODE, DATASET);                                                               \
    void TEST_NAME::do_run()

#define REGISTER_FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE)                        \
    class TEST_NAME : public arm_compute::test::framework::TestCase, public FIXTURE \
    {                                                                               \
    public:                                                                     \
        TEST_CASE_CONSTRUCTOR(TEST_NAME)                                            \
        FIXTURE_SETUP(FIXTURE)                                                      \
        FIXTURE_RUN(FIXTURE)                                                        \
        FIXTURE_TEARDOWN(FIXTURE)                                                   \
    };                                                                              \
    TEST_REGISTRAR(TEST_NAME, MODE)

#define REGISTER_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET)                                       \
    class TEST_NAME : public arm_compute::test::framework::DataTestCase<decltype(DATASET)::type>, public FIXTURE \
    {                                                                                                            \
    public:                                                                                                  \
        DATA_TEST_CASE_CONSTRUCTOR(TEST_NAME, DATASET)                                                           \
        FIXTURE_DATA_SETUP(FIXTURE)                                                                              \
        FIXTURE_RUN(FIXTURE)                                                                                     \
        FIXTURE_TEARDOWN(FIXTURE)                                                                                \
    };                                                                                                           \
    DATA_TEST_REGISTRAR(TEST_NAME, MODE, DATASET)
//
// TEST CASE MACROS END
//

#define ARM_COMPUTE_ASSERT_EQUAL(x, y)                                \
    do                                                                \
    {                                                                 \
        const auto &_x = (x);                                         \
        const auto &_y = (y);                                         \
        if(_x != _y)                                                  \
        {                                                             \
            std::stringstream msg;                                    \
            msg << "Assertion " << _x << " != " << _y << " failed.";  \
            throw arm_compute::test::framework::TestError(msg.str()); \
        }                                                             \
    } while(false)

#define ARM_COMPUTE_EXPECT_EQUAL(x, y)                                                        \
    do                                                                                        \
    {                                                                                         \
        const auto &_x = (x);                                                                 \
        const auto &_y = (y);                                                                 \
        if(_x != _y)                                                                          \
        {                                                                                     \
            std::stringstream msg;                                                            \
            msg << "Expectation " << _x << " != " << _y << " failed.";                        \
            arm_compute::test::framework::Framework::get().log_failed_expectation(msg.str()); \
        }                                                                                     \
    } while(false)
#endif /* ARM_COMPUTE_TEST_FRAMEWORK_MACROS */
