/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLTranspose.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/TransposeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(Transpose)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::U16), // Invalid shape
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::U8),  // Wrong data type
                                            TensorInfo(TensorShape(20U, 16U), 1, DataType::U32), // Valid
                                            TensorInfo(TensorShape(20U, 16U, 3U, 3U), 1, DataType::U16), // Transpose only first two dimensions
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(13U, 20U), 1, DataType::U32),
                                            TensorInfo(TensorShape(31U, 20U), 1, DataType::U16),
                                            TensorInfo(TensorShape(16U, 20U), 1, DataType::U32),
                                            TensorInfo(TensorShape(16U, 20U, 3U, 3U), 1, DataType::U16),
                                           })),
    framework::dataset::make("Expected", { false, false, true, true })),
    a_info, output_info, expected)
{
    // Lock tensors
    ARM_COMPUTE_EXPECT(bool(CLTranspose::validate(&a_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLTransposeFixture = TransposeValidationFixture<CLTensor, CLAccessor, CLTranspose, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLTransposeFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small1DShapes(), datasets::Small2DShapes()),
                                                                                                         framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLTransposeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large1DShapes(), datasets::Large2DShapes()),
                                                                                                       framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLargeHighDimensional,
                       CLTransposeFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(concat(concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                      datasets::Large5dShapes()),
                               framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLTransposeFixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small1DShapes(), datasets::Small2DShapes()),
                                                                                                          framework::dataset::make("DataType", DataType::U16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLTransposeFixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large1DShapes(), datasets::Large2DShapes()),
                                                                                                        framework::dataset::make("DataType", DataType::U16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLTransposeFixture<uint32_t>, framework::DatasetMode::PRECOMMIT, combine(concat(concat(framework::dataset::make("Shape", { TensorShape{ 1U, 5U }, TensorShape{ 4U, 5U }, TensorShape{ 3, 12 } }),
                                                                                                                        datasets::Small1DShapes()),
                                                                                                                 datasets::Small2DShapes()),
                                                                                                          framework::dataset::make("DataType", DataType::U32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallHighDimensional,
                       CLTransposeFixture<uint32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                               framework::dataset::make("DataType", DataType::U32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLTransposeFixture<uint32_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large1DShapes(), datasets::Large2DShapes()),
                                                                                                        framework::dataset::make("DataType", DataType::U32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U32

TEST_SUITE_END() // Transpose
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
