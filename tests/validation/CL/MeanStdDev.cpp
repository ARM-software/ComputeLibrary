/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLMeanStdDev.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/MeanStdDevFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_rel_high_error(0.05f);
RelativeTolerance<float> tolerance_rel_low_error(0.0005f);
AbsoluteTolerance<float> tolerance_rel_high_error_f32(0.01f);
AbsoluteTolerance<float> tolerance_rel_low_error_f32(0.00001f);
AbsoluteTolerance<float> tolerance_rel_high_error_f16(0.1f);
AbsoluteTolerance<float> tolerance_rel_low_error_f16(0.01f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(MeanStdDev)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),    // Wrong input data type
                                                       TensorInfo(TensorShape(16U, 5U, 16U), 1, DataType::U8), // Invalid shape
                                                       TensorInfo(TensorShape(16U, 16U), 1, DataType::U8),     // Valid
                                                     }),
               framework::dataset::make("Expected", { false, false, true })),
               input_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLMeanStdDev::validate(&input_info.clone()->set_is_resizable(false), nullptr, nullptr)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLMeanStdDevFixture = MeanStdDevValidationFixture<CLTensor, CLAccessor, CLMeanStdDev, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMeanStdDevFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                        DataType::U8)))
{
    // Validate mean output
    validate(_target.first, _reference.first);

    // Validate std_dev output
    validate(_target.second, _reference.second, tolerance_rel_high_error);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLMeanStdDevFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("DataType",
                                                                                                        DataType::U8)))
{
    // Validate mean output
    validate(_target.first, _reference.first, tolerance_rel_low_error);

    // Validate std_dev output
    validate(_target.second, _reference.second, tolerance_rel_high_error);
}
TEST_SUITE_END() // U8

TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMeanStdDevFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                     DataType::F16)))
{
    // Validate mean output
    validate(_target.first, _reference.first, tolerance_rel_low_error_f16);

    // Validate std_dev output
    validate(_target.second, _reference.second, tolerance_rel_high_error_f16);
}
TEST_SUITE_END() // F16

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMeanStdDevFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                      DataType::F32)))
{
    // Validate mean output
    validate(_target.first, _reference.first, tolerance_rel_low_error_f32);

    // Validate std_dev output
    validate(_target.second, _reference.second, tolerance_rel_high_error_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLMeanStdDevFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("DataType",
                                                                                                      DataType::F32)))
{
    // Validate mean output
    validate(_target.first, _reference.first, tolerance_rel_low_error_f32);

    // Validate std_dev output
    validate(_target.second, _reference.second, tolerance_rel_high_error_f32);
}
TEST_SUITE_END() // F32

TEST_SUITE_END() // MeanStdDev
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
