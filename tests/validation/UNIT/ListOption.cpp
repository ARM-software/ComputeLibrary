/*
 * SPDX-FileCopyrightText: 2026 Yusuf Efe
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
#include "utils/command_line/ListOption.h"

#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

#include <string>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(ListOption)

TEST_CASE(PreserveStringWhitespace, framework::DatasetMode::ALL)
{
    utils::ListOption<std::string> option("example_args");
    const std::vector<std::string> expected{"--image=/tmp/test images/sample.ppm", "--threads=2"};
    ARM_COMPUTE_EXPECT(option.parse("--image=/tmp/test images/sample.ppm,--threads=2"), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(option.value() == expected, framework::LogLevel::ERRORS);
}

TEST_CASE(PreserveLeadingTrailingAndTabWhitespace, framework::DatasetMode::ALL)
{
    utils::ListOption<std::string> option("values");
    const std::vector<std::string> expected{"  first value  ", "second\tvalue", " "};
    ARM_COMPUTE_EXPECT(option.parse("  first value  ,second\tvalue, "), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(option.value() == expected, framework::LogLevel::ERRORS);
}

TEST_CASE(AppendStringValues, framework::DatasetMode::ALL)
{
    utils::ListOption<std::string> option("values", {"default"});
    const std::vector<std::string> expected{"default", "first", "second", "third value"};
    ARM_COMPUTE_EXPECT(option.parse("first,second"), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(option.parse("third value"), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(option.value() == expected, framework::LogLevel::ERRORS);
}

TEST_CASE(EmptyStringInput, framework::DatasetMode::ALL)
{
    utils::ListOption<std::string> option("values");
    ARM_COMPUTE_EXPECT(option.parse(""), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(option.value().empty(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(option.parse("value,"), framework::LogLevel::ERRORS);
    const std::vector<std::string> expected{"value"};
    ARM_COMPUTE_EXPECT(option.value() == expected, framework::LogLevel::ERRORS);
}

TEST_CASE(RejectEmptyStringItem, framework::DatasetMode::ALL)
{
    utils::ListOption<std::string> option("values");
    ARM_COMPUTE_EXPECT(!option.parse("first,,second"), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!option.is_set(), framework::LogLevel::ERRORS);
    const std::vector<std::string> expected{"first", "second"};
    ARM_COMPUTE_EXPECT(option.value() == expected, framework::LogLevel::ERRORS);
}

TEST_CASE(ParseIntegerValues, framework::DatasetMode::ALL)
{
    utils::ListOption<int> option("values");
    const std::vector<int> expected{1, -2, 3};
    ARM_COMPUTE_EXPECT(option.parse("1, -2,3"), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(option.value() == expected, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!option.parse("invalid"), framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // ListOption
TEST_SUITE_END() // UNIT
} // namespace validation
} // namespace test
} // namespace arm_compute
