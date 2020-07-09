/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLElementWiseUnaryLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ElementWiseUnaryFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_fp32(0.000001f);
RelativeTolerance<float> tolerance_fp16(0.001f);
} // namespace
TEST_SUITE(CL)
TEST_SUITE(RsqrtLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Valid
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { false, true, false })),
               input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLRsqrtLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
template <typename T>
using CLRsqrtLayerFixture = RsqrtValidationFixture<CLTensor, CLAccessor, CLRsqrtLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRsqrtLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                       DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLRsqrtLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                     DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16);
}

TEST_SUITE_END() // FP16
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRsqrtLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                        DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLRsqrtLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                      DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // RsqrtLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
