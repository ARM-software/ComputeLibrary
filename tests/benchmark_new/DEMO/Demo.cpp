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
#include "framework/Macros.h"
#include "tests/DatasetManager.h"

#include <vector>

namespace arm_compute
{
namespace test
{
TEST_SUITE(DEMO)

#if 0
TEST_CASE(NoArgs)
{
    ARM_COMPUTE_EXPECT_EQUAL(6, 5);
}

DATA_TEST_CASE(And, framework::dataset::make("Foo", { 5, 7, 9 }), int num)
{
    ARM_COMPUTE_EXPECT_EQUAL(6, num);
}

DATA_TEST_CASE(And2, framework::dataset::zip(framework::dataset::make("Foo", 2), framework::dataset::make("Bar", 3.f)), int num, float count)
{
}

DATA_TEST_CASE(And3, framework::DatasetManager::get().shapesDataset(), arm_compute::TensorShape num)
{
}

DATA_TEST_CASE(And4, framework::dataset::zip(framework::dataset::make("Zip", std::vector<int> { 2, 3, 4 }), framework::dataset::make("Bar", std::vector<int> { -2, -3, -4 })), int num, float count)
{
    ARM_COMPUTE_ASSERT_EQUAL(num, count);
}

DATA_TEST_CASE(And5, framework::dataset::make("Foo", { 2, 3, 4 }), int num)
{
}

DATA_TEST_CASE(And6, framework::dataset::combine(framework::dataset::make("Foo", std::vector<int> { 2, 3, 4 }), framework::dataset::make("Bar", std::vector<int> { -2, -3, -4 })), int num, int count)
{
    ARM_COMPUTE_EXPECT_EQUAL(num, count);
}

DATA_TEST_CASE(And7, framework::dataset::combine(framework::dataset::combine(framework::dataset::make("Foo", std::vector<int> { 2, 3, 4 }), framework::dataset::make("Bar", std::vector<int> { -2, -3, -4 })),
                                                 framework::dataset::make("FooBar", std::vector<int> { 5, 6 })),
               int num,
               int count, int asd)
{
}

DATA_TEST_CASE(And8, framework::dataset::concat(framework::dataset::make("Foo", std::vector<int> { 2, 3, 4 }), framework::dataset::make("Bar", std::vector<int> { -2, -3, -4 })), int num)
{
}

class MyFixture : public framework::Fixture
{
public:
    MyFixture()
    {
        std::cout << "Created fixture!!!\n";
    }

    MyFixture(const MyFixture &) = default;
    MyFixture(MyFixture &&)      = default;
    MyFixture &operator=(const MyFixture &) = default;
    MyFixture &operator=(MyFixture &&) = default;

    void setup()
    {
        std::cout << "Set up fixture!!!\n";
    }

    void run()
    {
        std::cout << "Run fixture\n";
    }

    void teardown()
    {
        std::cout << "Tear down fixture!!!\n";
    }

    ~MyFixture()
    {
        std::cout << "Destroyed fixture!!!\n";
    }
};

class MyDataFixture : public framework::Fixture
{
public:
    MyDataFixture()
    {
        std::cout << "Created data fixture!!!\n";
    }

    MyDataFixture(const MyDataFixture &) = default;
    MyDataFixture(MyDataFixture &&)      = default;
    MyDataFixture &operator=(const MyDataFixture &) = default;
    MyDataFixture &operator=(MyDataFixture &&) = default;

    void setup(int num)
    {
        _num = num;
        std::cout << "Set up fixture with num = " << _num << "\n";
    }

    void run()
    {
        std::cout << "Run fixture\n";
    }

    ~MyDataFixture()
    {
        std::cout << "Destroyed data fixture!!!\n";
    }

protected:
    int _num{};
};

FIXTURE_TEST_CASE(And11, MyFixture)
{
    std::cout << "Running fixture test!!!\n";
}

FIXTURE_DATA_TEST_CASE(And12, MyDataFixture, framework::dataset::make("Foo", { 2, 3, 4 }))
{
    std::cout << "Running fixture test with value " << _num << "!!!\n";
}

REGISTER_FIXTURE_TEST_CASE(And13, MyFixture);
REGISTER_FIXTURE_DATA_TEST_CASE(And14, MyDataFixture, framework::dataset::make("Foo", { 2, 3, 4 }));
#endif /* 0 */

TEST_SUITE_END()
} // namespace test
} // namespace arm_compute
