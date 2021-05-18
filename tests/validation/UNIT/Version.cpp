/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/AclVersion.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_CASE(Version, framework::DatasetMode::ALL)
{
    const auto ver = AclVersionInfo();
    ARM_COMPUTE_EXPECT(ver->major == ARM_COMPUTE_LIBRARY_VERSION_MAJOR, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(ver->minor == ARM_COMPUTE_LIBRARY_VERSION_MINOR, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(ver->patch == ARM_COMPUTE_LIBRARY_VERSION_PATCH, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!std::string(ver->build_info).empty(), framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // UNIT
} // namespace validation
} // namespace test
} // namespace arm_compute
