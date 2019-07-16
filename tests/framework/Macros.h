/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "Framework.h"
#include "Registrars.h"
#include "TestCase.h"

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
// HELPER MACROS
//

#define CONCAT(ARG0, ARG1) ARG0##ARG1

#define VARIADIC_SIZE_IMPL(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, size, ...) size
#define VARIADIC_SIZE(...) VARIADIC_SIZE_IMPL(__VA_ARGS__, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define JOIN_PARAM1(OP, param) OP(0, param)
#define JOIN_PARAM2(OP, param, ...) \
    OP(1, param)                    \
    , JOIN_PARAM1(OP, __VA_ARGS__)
#define JOIN_PARAM3(OP, param, ...) \
    OP(2, param)                    \
    , JOIN_PARAM2(OP, __VA_ARGS__)
#define JOIN_PARAM4(OP, param, ...) \
    OP(3, param)                    \
    , JOIN_PARAM3(OP, __VA_ARGS__)
#define JOIN_PARAM5(OP, param, ...) \
    OP(4, param)                    \
    , JOIN_PARAM4(OP, __VA_ARGS__)
#define JOIN_PARAM6(OP, param, ...) \
    OP(5, param)                    \
    , JOIN_PARAM5(OP, __VA_ARGS__)
#define JOIN_PARAM7(OP, param, ...) \
    OP(6, param)                    \
    , JOIN_PARAM6(OP, __VA_ARGS__)
#define JOIN_PARAM8(OP, param, ...) \
    OP(7, param)                    \
    , JOIN_PARAM7(OP, __VA_ARGS__)
#define JOIN_PARAM9(OP, param, ...) \
    OP(8, param)                    \
    , JOIN_PARAM8(OP, __VA_ARGS__)
#define JOIN_PARAM10(OP, param, ...) \
    OP(9, param)                     \
    , JOIN_PARAM9(OP, __VA_ARGS__)
#define JOIN_PARAM11(OP, param, ...) \
    OP(10, param)                    \
    , JOIN_PARAM10(OP, __VA_ARGS__)
#define JOIN_PARAM12(OP, param, ...) \
    OP(11, param)                    \
    , JOIN_PARAM11(OP, __VA_ARGS__)
#define JOIN_PARAM13(OP, param, ...) \
    OP(12, param)                    \
    , JOIN_PARAM12(OP, __VA_ARGS__)
#define JOIN_PARAM(OP, NUM, ...) \
    CONCAT(JOIN_PARAM, NUM)      \
    (OP, __VA_ARGS__)

#define MAKE_TYPE_PARAM(i, name) typename T##i
#define MAKE_ARG_PARAM(i, name) const T##i &name
#define MAKE_TYPE_PARAMS(...) JOIN_PARAM(MAKE_TYPE_PARAM, VARIADIC_SIZE(__VA_ARGS__), __VA_ARGS__)
#define MAKE_ARG_PARAMS(...) JOIN_PARAM(MAKE_ARG_PARAM, VARIADIC_SIZE(__VA_ARGS__), __VA_ARGS__)

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
#define FIXTURE_DATA_SETUP(FIXTURE)                 \
    void do_setup() override                        \
    {                                               \
        apply(this, &FIXTURE::setup<As...>, _data); \
    }
#define FIXTURE_RUN(FIXTURE) \
    void do_run() override   \
    {                        \
        FIXTURE::run();      \
    }
#define FIXTURE_SYNC(FIXTURE) \
    void do_sync() override   \
    {                         \
        FIXTURE::sync();      \
    }
#define FIXTURE_TEARDOWN(FIXTURE) \
    void do_teardown() override   \
    {                             \
        FIXTURE::teardown();      \
    }
#define TEST_REGISTRAR(TEST_NAME, MODE, STATUS)                                               \
    static arm_compute::test::framework::detail::TestCaseRegistrar<TEST_NAME> TEST_NAME##_reg \
    {                                                                                         \
        #TEST_NAME, MODE, STATUS                                                              \
    }
#define DATA_TEST_REGISTRAR(TEST_NAME, MODE, STATUS, DATASET)                                                          \
    static arm_compute::test::framework::detail::TestCaseRegistrar<TEST_NAME<decltype(DATASET)::type>> TEST_NAME##_reg \
    {                                                                                                                  \
        #TEST_NAME, MODE, STATUS, DATASET                                                                              \
    }

#define TEST_CASE_IMPL(TEST_NAME, MODE, STATUS)                     \
    class TEST_NAME : public arm_compute::test::framework::TestCase \
    {                                                               \
    public:                                                     \
        TEST_CASE_CONSTRUCTOR(TEST_NAME)                            \
        void do_run() override;                                     \
    };                                                              \
    TEST_REGISTRAR(TEST_NAME, MODE, STATUS);                        \
    void TEST_NAME::do_run()

#define TEST_CASE(TEST_NAME, MODE) \
    TEST_CASE_IMPL(TEST_NAME, MODE, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE)
#define EXPECTED_FAILURE_TEST_CASE(TEST_NAME, MODE) \
    TEST_CASE_IMPL(TEST_NAME, MODE, arm_compute::test::framework::TestCaseFactory::Status::EXPECTED_FAILURE)
#define DISABLED_TEST_CASE(TEST_NAME, MODE) \
    TEST_CASE_IMPL(TEST_NAME, MODE, arm_compute::test::framework::TestCaseFactory::Status::DISABLED)

#define DATA_TEST_CASE_IMPL(TEST_NAME, MODE, STATUS, DATASET, ...)                                                  \
    template <typename T>                                                                                           \
    class TEST_NAME;                                                                                                \
    template <typename... As>                                                                                       \
    class TEST_NAME<std::tuple<As...>> : public arm_compute::test::framework::DataTestCase<decltype(DATASET)::type> \
    {                                                                                                               \
    public:                                                                                                     \
        DATA_TEST_CASE_CONSTRUCTOR(TEST_NAME, DATASET)                                                              \
        void do_run() override                                                                                      \
        {                                                                                                           \
            arm_compute::test::framework::apply(this, &TEST_NAME::run<As...>, _data);                               \
        }                                                                                                           \
        template <MAKE_TYPE_PARAMS(__VA_ARGS__)>                                                                    \
        void run(MAKE_ARG_PARAMS(__VA_ARGS__));                                                                     \
    };                                                                                                              \
    DATA_TEST_REGISTRAR(TEST_NAME, MODE, STATUS, DATASET);                                                          \
    template <typename... As>                                                                                       \
    template <MAKE_TYPE_PARAMS(__VA_ARGS__)>                                                                        \
    void TEST_NAME<std::tuple<As...>>::run(MAKE_ARG_PARAMS(__VA_ARGS__))

#define DATA_TEST_CASE(TEST_NAME, MODE, DATASET, ...) \
    DATA_TEST_CASE_IMPL(TEST_NAME, MODE, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE, DATASET, __VA_ARGS__)
#define EXPECTED_FAILURE_DATA_TEST_CASE(TEST_NAME, MODE, DATASET, ...) \
    DATA_TEST_CASE_IMPL(TEST_NAME, MODE, arm_compute::test::framework::TestCaseFactory::Status::EXPECTED_FAILURE, DATASET, __VA_ARGS__)
#define DISABLED_DATA_TEST_CASE(TEST_NAME, MODE, DATASET, ...) \
    DATA_TEST_CASE_IMPL(TEST_NAME, MODE, arm_compute::test::framework::TestCaseFactory::Status::DISABLED, DATASET, __VA_ARGS__)

#define FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, STATUS)                    \
    class TEST_NAME : public arm_compute::test::framework::TestCase, public FIXTURE \
    {                                                                               \
    public:                                                                     \
        TEST_CASE_CONSTRUCTOR(TEST_NAME)                                            \
        FIXTURE_SETUP(FIXTURE)                                                      \
        void do_run() override;                                                     \
        FIXTURE_TEARDOWN(FIXTURE)                                                   \
    };                                                                              \
    TEST_REGISTRAR(TEST_NAME, MODE, STATUS);                                        \
    void TEST_NAME::do_run()

#define FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE) \
    FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE)
#define EXPECTED_FAILURE_FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE) \
    FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::EXPECTED_FAILURE)
#define DISABLED_FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE) \
    FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::DISABLED)

#define FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, STATUS, DATASET)                                                      \
    template <typename T>                                                                                                           \
    class TEST_NAME;                                                                                                                \
    template <typename... As>                                                                                                       \
    class TEST_NAME<std::tuple<As...>> : public arm_compute::test::framework::DataTestCase<decltype(DATASET)::type>, public FIXTURE \
    {                                                                                                                               \
    public:                                                                                                                     \
        DATA_TEST_CASE_CONSTRUCTOR(TEST_NAME, DATASET)                                                                              \
        FIXTURE_DATA_SETUP(FIXTURE)                                                                                                 \
        void do_run() override;                                                                                                     \
        FIXTURE_TEARDOWN(FIXTURE)                                                                                                   \
    };                                                                                                                              \
    DATA_TEST_REGISTRAR(TEST_NAME, MODE, STATUS, DATASET);                                                                          \
    template <typename... As>                                                                                                       \
    void TEST_NAME<std::tuple<As...>>::do_run()

#define FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET) \
    FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE, DATASET)
#define EXPECTED_FAILURE_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET) \
    FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::EXPECTED_FAILURE, DATASET)
#define DISABLED_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET) \
    FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::DISABLED, DATASET)

#define REGISTER_FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, STATUS)           \
    class TEST_NAME : public arm_compute::test::framework::TestCase, public FIXTURE \
    {                                                                               \
    public:                                                                     \
        TEST_CASE_CONSTRUCTOR(TEST_NAME)                                            \
        FIXTURE_SETUP(FIXTURE)                                                      \
        FIXTURE_RUN(FIXTURE)                                                        \
        FIXTURE_SYNC(FIXTURE)                                                       \
        FIXTURE_TEARDOWN(FIXTURE)                                                   \
    };                                                                              \
    TEST_REGISTRAR(TEST_NAME, MODE, STATUS)

#define REGISTER_FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE) \
    REGISTER_FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE)
#define EXPECTED_FAILURE_REGISTER_FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE) \
    REGISTER_FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::EXPECTED_FAILURE)
#define DISABLED_REGISTER_FIXTURE_TEST_CASE(TEST_NAME, FIXTURE, MODE) \
    REGISTER_FIXTURE_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::DISABLED)

#define REGISTER_FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, STATUS, DATASET)                                             \
    template <typename T>                                                                                                           \
    class TEST_NAME;                                                                                                                \
    template <typename... As>                                                                                                       \
    class TEST_NAME<std::tuple<As...>> : public arm_compute::test::framework::DataTestCase<decltype(DATASET)::type>, public FIXTURE \
    {                                                                                                                               \
    public:                                                                                                                     \
        DATA_TEST_CASE_CONSTRUCTOR(TEST_NAME, DATASET)                                                                              \
        FIXTURE_DATA_SETUP(FIXTURE)                                                                                                 \
        FIXTURE_RUN(FIXTURE)                                                                                                        \
        FIXTURE_SYNC(FIXTURE)                                                                                                       \
        FIXTURE_TEARDOWN(FIXTURE)                                                                                                   \
    };                                                                                                                              \
    DATA_TEST_REGISTRAR(TEST_NAME, MODE, STATUS, DATASET)

#define REGISTER_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET) \
    REGISTER_FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE, DATASET)
#define EXPECTED_FAILURE_REGISTER_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET) \
    REGISTER_FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::EXPECTED_FAILURE, DATASET)
#define DISABLED_REGISTER_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, DATASET) \
    REGISTER_FIXTURE_DATA_TEST_CASE_IMPL(TEST_NAME, FIXTURE, MODE, arm_compute::test::framework::TestCaseFactory::Status::DISABLED, DATASET)
//
// TEST CASE MACROS END
//
#endif /* ARM_COMPUTE_TEST_FRAMEWORK_MACROS */