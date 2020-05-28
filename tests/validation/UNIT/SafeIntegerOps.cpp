/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/utils/math/SafeOps.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(SafeIntegerOps)

TEST_CASE(IntegerOverflowAdd, framework::DatasetMode::ALL)
{
    int32_t val_a  = 0x7FFFFFFF;
    int32_t val_b  = 0xFF;
    int32_t result = utils::math::safe_integer_add(val_a, val_b);

    // Check overflow
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::max(), framework::LogLevel::ERRORS);

    val_a  = 0x8000FC24;
    val_b  = 0x80000024;
    result = utils::math::safe_integer_add(val_a, val_b);

    // Check underflow
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::min(), framework::LogLevel::ERRORS);
}

TEST_CASE(IntegerOverflowSub, framework::DatasetMode::ALL)
{
    int32_t val_a  = 0x7FFFFFFF;
    int32_t val_b  = 0x8000FC24;
    int32_t result = utils::math::safe_integer_sub(val_a, val_b);

    // Check overflow
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::max(), framework::LogLevel::ERRORS);

    val_a  = 0x80000024;
    val_b  = 0x7FFFFFFF;
    result = utils::math::safe_integer_sub(val_a, val_b);

    // Check underflow
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::min(), framework::LogLevel::ERRORS);
}

TEST_CASE(IntegerOverflowMul, framework::DatasetMode::ALL)
{
    int32_t val_a  = 0xFFFFFFFF;
    int32_t val_b  = 0x80000000;
    int32_t result = utils::math::safe_integer_mul(val_a, val_b);

    // Check overflow with -1
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::min(), framework::LogLevel::ERRORS);

    val_a  = 0x80000000;
    val_b  = 0xFFFFFFFF;
    result = utils::math::safe_integer_mul(val_a, val_b);

    // Check overflow with -1
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::min(), framework::LogLevel::ERRORS);

    // Check overflow
    val_a  = 0x7000FC24;
    val_b  = 0x70000024;
    result = utils::math::safe_integer_mul(val_a, val_b);
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::max(), framework::LogLevel::ERRORS);

    // Check underflow
    val_a  = 0x7000FC24;
    val_b  = 0xF0000024;
    result = utils::math::safe_integer_mul(val_a, val_b);
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::min(), framework::LogLevel::ERRORS);
}

TEST_CASE(IntegerOverflowDiv, framework::DatasetMode::ALL)
{
    int32_t val_a  = std::numeric_limits<int32_t>::min();
    int32_t val_b  = 0xFFFFFFFF;
    int32_t result = utils::math::safe_integer_div(val_a, val_b);

    // Check overflow
    ARM_COMPUTE_EXPECT(result == std::numeric_limits<int32_t>::min(), framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // SafeIntegerOps
TEST_SUITE_END() // UNIT
} // namespace validation
} // namespace test
} // namespace arm_compute
