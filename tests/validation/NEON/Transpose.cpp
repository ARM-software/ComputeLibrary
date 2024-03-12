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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
TEST_SUITE(NEON)
TEST_SUITE(Transpose)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::U16), // Invalid shape
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::U8),  // Wrong data type
                                            TensorInfo(TensorShape(20U, 16U), 1, DataType::U16),
                                            TensorInfo(TensorShape(20U, 16U), 1, DataType::U32),
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::U16),
                                            TensorInfo(TensorShape(31U, 20U), 1, DataType::U16),
                                            TensorInfo(TensorShape(16U, 20U), 1, DataType::U16),
                                            TensorInfo(TensorShape(16U, 20U), 1, DataType::U32),
                                           })),
    framework::dataset::make("Expected", { false, false, true, true })),
    a_info, output_info, expected)
{
    // Lock tensors
    Status status =  NETranspose::validate(&a_info.clone()->set_is_resizable(false),
                                           &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NETransposeFixture = TransposeValidationFixture<Tensor, Accessor, NETranspose, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NETransposeFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small1DShapes(), datasets::Small2DShapes()),
                                                                                                         framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETransposeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large1DShapes(), datasets::Large2DShapes()),
                                                                                                       framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall, NETransposeFixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small1DShapes(), datasets::Small2DShapes()),
                                                                                                          framework::dataset::make("DataType", DataType::U16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETransposeFixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large1DShapes(), datasets::Large2DShapes()),
                                                                                                        framework::dataset::make("DataType", DataType::U16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmall, NETransposeFixture<uint32_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small1DShapes(), datasets::Small2DShapes()),
                                                                                                          framework::dataset::make("DataType", DataType::U32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETransposeFixture<uint32_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large1DShapes(), datasets::Large2DShapes()),
                                                                                                        framework::dataset::make("DataType", DataType::U32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
