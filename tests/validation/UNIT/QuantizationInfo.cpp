/*
 * Copyright (c) 2026 Arm Limited.
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
#include "arm_compute/core/QuantizationInfo.h"

#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(QuantizationInfo)

TEST_CASE(EquivalentUniformImplicitAndExplicitZeroOffset, framework::DatasetMode::ALL)
{
    const QuantizationInfo implicit_zero(1.0f);
    const QuantizationInfo explicit_zero(1.0f, 0);

    ARM_COMPUTE_EXPECT(implicit_zero == explicit_zero, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!(implicit_zero != explicit_zero), framework::LogLevel::ERRORS);
}

TEST_CASE(DifferentUniformQInfoCompareDifferent, framework::DatasetMode::ALL)
{
    const QuantizationInfo unit_scale(1.0f);
    const QuantizationInfo half_scale(0.5f, 0);
    const QuantizationInfo offset_one(1.0f, 1);
    const QuantizationInfo offset_zero(1.0f, 0);

    ARM_COMPUTE_EXPECT(unit_scale != half_scale, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(offset_one != offset_zero, framework::LogLevel::ERRORS);
}

TEST_CASE(PerChannelQInfoUsesStrictVectorEquality, framework::DatasetMode::ALL)
{
    const QuantizationInfo matching_per_channel(std::vector<float>{1.0f, 2.0f});
    const QuantizationInfo same_per_channel(std::vector<float>{1.0f, 2.0f});
    const QuantizationInfo different_per_channel(std::vector<float>{1.0f, 3.0f});

    ARM_COMPUTE_EXPECT(matching_per_channel == same_per_channel, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(matching_per_channel != different_per_channel, framework::LogLevel::ERRORS);
}

TEST_CASE(PerChannelQInfoDoesNotMatchUniformQInfo, framework::DatasetMode::ALL)
{
    const QuantizationInfo per_channel(std::vector<float>{1.0f, 2.0f});
    const QuantizationInfo uniform(1.0f);

    ARM_COMPUTE_EXPECT(per_channel != uniform, framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // QuantizationInfo
TEST_SUITE_END() // UNIT
} // namespace validation
} // namespace test
} // namespace arm_compute
