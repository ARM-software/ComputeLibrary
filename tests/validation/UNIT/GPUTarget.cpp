/*
 * Copyright (c) 2018-2024 Arm Limited.
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
TEST_SUITE(GPUTarget)

TEST_CASE(GetGPUTargetFromName, framework::DatasetMode::ALL)
{
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-T000") == GPUTarget::MIDGARD, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-T600") == GPUTarget::T600, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-T700") == GPUTarget::T700, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-T800") == GPUTarget::T800, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G71") == GPUTarget::G71, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G72") == GPUTarget::G72, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G51") == GPUTarget::G51, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G51BIG") == GPUTarget::G51BIG, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G51LIT") == GPUTarget::G51LIT, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G31") == GPUTarget::G31, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G76") == GPUTarget::G76, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G76 r0p0") == GPUTarget::G76, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G52") == GPUTarget::G52, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G52LIT") == GPUTarget::G52LIT, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G77") == GPUTarget::G77, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G57") == GPUTarget::G57, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G78") == GPUTarget::G78, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G68") == GPUTarget::G68, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G78AE") == GPUTarget::G78AE, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G710") == GPUTarget::G710, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G610") == GPUTarget::G610, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G510") == GPUTarget::G510, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G310") == GPUTarget::G310, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G715") == GPUTarget::G715, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G615") == GPUTarget::G615, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G720") == GPUTarget::G720, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G620") == GPUTarget::G620, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(get_target_from_name("Mali-G10000") == GPUTarget::FIFTHGEN, framework::LogLevel::ERRORS); // Unknown Gpu
}

TEST_CASE(GPUTargetIsIn, framework::DatasetMode::ALL)
{
    ARM_COMPUTE_EXPECT(!gpu_target_is_in(GPUTarget::G71, GPUTarget::T600, GPUTarget::T800, GPUTarget::G72), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(gpu_target_is_in(GPUTarget::G71, GPUTarget::T600, GPUTarget::T800, GPUTarget::G71), framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // CLHelpers
TEST_SUITE_END() // UNIT
} // namespace validation
} // namespace test
} // namespace arm_compute
