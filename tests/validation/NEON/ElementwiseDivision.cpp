/*
 * Copyright (c) 2019-2021, 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/ElementwiseOperationsFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
RelativeTolerance<float> tolerance_fp32(0.000001f);
AbsoluteTolerance<int>   tolerance_zero_s32(0); // Tolerance for S32 division

/** Input data sets **/
const auto ElementwiseDivisionS32Dataset =
    combine(make("DataType", DataType::S32), make("DataType", DataType::S32), make("DataType", DataType::S32));
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<half> tolerance_fp16(static_cast<half>(0.01f));
const auto              ElementwiseDivisionFP16Dataset =
    combine(make("DataType", DataType::F16), make("DataType", DataType::F16), make("DataType", DataType::F16));
#endif /* ARM_COMPUTE_ENABLE_FP16 */
const auto ElementwiseDivisionFP32Dataset =
    combine(make("DataType", DataType::F32), make("DataType", DataType::F32), make("DataType", DataType::F32));
const auto InPlaceDataSet    = make("InPlace", {false, true});
const auto OutOfPlaceDataSet = make("InPlace", {false});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ElementwiseDivision)

template <typename T>
using NEElementwiseDivisionFixture = ArithmeticDivisionValidationFixture<Tensor, Accessor, NEElementwiseDivision, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     }),
               make("Expected", { true, true, true, false, false})
               ),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEElementwiseDivision::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

// Test test cases will execute the function with dynamic-stated shapes
// Since other elementwise operations share the same kernel, this tests are added only here.
// Also, only FP32 is tested since data type doesn't/shouldn't matter with dynamic shapes.
TEST_SUITE(DynamicShape)
template <typename T>
using CpuElementwiseDivisionDynamicShapeFixture =
    ArithmeticDivisionDynamicShapeValidationFixture<Tensor, Accessor, NEElementwiseDivision, T>;

template <typename T>
using CpuElementwiseDivisionBroadcastDynamicShapeFixture =
    ArithmeticDivisionBroadcastDynamicShapeValidationFixture<Tensor, Accessor, NEElementwiseDivision, T>;

TEST_SUITE(F32)

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuElementwiseDivisionDynamicShapeFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseDivisionFP32Dataset, InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       CpuElementwiseDivisionBroadcastDynamicShapeFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesBroadcast(), ElementwiseDivisionFP32Dataset, OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}

TEST_SUITE_END() // F32
TEST_SUITE_END() // DynamicShape

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEElementwiseDivisionFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseDivisionFP16Dataset, InPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_fp16, 0.01);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // F16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEElementwiseDivisionFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseDivisionFP32Dataset, InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}

template <typename T>
using NEElementwiseDivisionBroadcastFixture =
    ArithmeticDivisionBroadcastValidationFixture<Tensor, Accessor, NEElementwiseDivision, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEElementwiseDivisionBroadcastFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesBroadcast(), ElementwiseDivisionFP32Dataset, OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
FIXTURE_DATA_TEST_CASE(RunTinyBroadcastInPlace,
                       NEElementwiseDivisionBroadcastFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::TinyShapesBroadcastInplace(), ElementwiseDivisionFP32Dataset, InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEElementwiseDivisionFixture<int32_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseDivisionS32Dataset, InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_zero_s32);
}
TEST_SUITE_END() // S32
TEST_SUITE_END() // Integer

TEST_SUITE_END() // ElementwiseDivision
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
