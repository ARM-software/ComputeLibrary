/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "tests/CL/Helper.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WeightsReshapeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(WeightsReshape)

using CLWeightsReshape = CLSynthetizeFunction<CLWeightsReshapeKernel>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::U8),      // Unsupported data type
                                                       TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),     // Mismatching data type
                                                       TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::QASYMM8), // Bias not supported with QASYMM8
                                                       TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                                     }),
               framework::dataset::make("BiasesInfo", { TensorInfo(TensorShape(4U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(4U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(4U), 1, DataType::QASYMM8),
                                                        TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                      })),
               framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(4U, 19U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(4U, 19U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(4U, 19U), 1, DataType::QASYMM8),
                                                        TensorInfo(TensorShape(4U, 19U), 1, DataType::F32),
                                                      })),
               framework::dataset::make("Expected", { false, false, false, true })),
               input_info, biases_info, output_info, expected)
{
    bool status = bool(CLWeightsReshape::validate(&input_info, &biases_info, &output_info));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLWeightsReshapeFixture = WeightsReshapeValidationFixture<CLTensor, CLAccessor, CLWeightsReshape, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWeightsReshapeFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::GroupedWeightsSmallShapes(), framework::dataset::make("DataType",
                                                                                                                      DataType::F32)),
                                                                                                              framework::dataset::make("HasBias", { true, false })),
                                                                                                      framework::dataset::make("NumGroups", { 1, 2, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWeightsReshapeFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::GroupedWeightsLargeShapes(), framework::dataset::make("DataType",
                                                                                                                  DataType::F32)),
                                                                                                                  framework::dataset::make("HasBias", { true, false })),
                                                                                                          framework::dataset::make("NumGroups", { 1, 2, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWeightsReshapeFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::GroupedWeightsSmallShapes(), framework::dataset::make("DataType",
                                                                                                                     DataType::F16)),
                                                                                                             framework::dataset::make("HasBias", { true, false })),
                                                                                                     framework::dataset::make("NumGroups", { 1, 2, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWeightsReshapeFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::GroupedWeightsLargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::F16)),
                                                                                                                 framework::dataset::make("HasBias", { true, false })),
                                                                                                         framework::dataset::make("NumGroups", { 1, 2, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWeightsReshapeFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::GroupedWeightsSmallShapes(), framework::dataset::make("DataType",
                                                                                                                        DataType::QASYMM8)),
                                                                                                                framework::dataset::make("HasBias", { false })),
                                                                                                        framework::dataset::make("NumGroups", { 1, 2, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWeightsReshapeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::GroupedWeightsLargeShapes(), framework::dataset::make("DataType",
                                                                                                                    DataType::QASYMM8)),
                                                                                                                    framework::dataset::make("HasBias", { false })),
                                                                                                            framework::dataset::make("NumGroups", { 1, 2, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
