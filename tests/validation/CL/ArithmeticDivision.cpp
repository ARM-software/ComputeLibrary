/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ElementwiseOperationsFixture.h"

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

/** Input data sets **/
const auto ArithmeticDivisionFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                                   framework::dataset::make("DataType", DataType::F16));
const auto ArithmeticDivisionFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                                   framework::dataset::make("DataType", DataType::F32));
const auto ArithmeticDivisionS32Dataset = combine(combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::S32)),
                                                  framework::dataset::make("DataType", DataType::S32));
const auto EmptyActivationFunctionsDataset = framework::dataset::make("ActivationInfo", { ActivationLayerInfo() });
const auto ActivationFunctionsDataset      = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.75f, 0.25f)
});
const auto InPlaceDataSet    = framework::dataset::make("InPlace", { false, true });
const auto OutOfPlaceDataSet = framework::dataset::make("InPlace", { false });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ArithmeticDivision)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),      // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, false, false, false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLArithmeticDivision::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

using CLArithmeticDivisionIntegerFixture = ArithmeticDivisionValidationIntegerFixture<CLTensor, CLAccessor, CLArithmeticDivision, int>;

TEST_SUITE(Integer)
TEST_SUITE(S32)

FIXTURE_DATA_TEST_CASE(RunSmallInteger, CLArithmeticDivisionIntegerFixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), ArithmeticDivisionS32Dataset),
                                                                                                                       EmptyActivationFunctionsDataset),
                                                                                                                       InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunIntegerWithActivation, CLArithmeticDivisionIntegerFixture, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), ArithmeticDivisionS32Dataset),
                       ActivationFunctionsDataset),
                       InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLArithmeticDivisionFloatFixture = ArithmeticDivisionValidationFloatFixture<CLTensor, CLAccessor, CLArithmeticDivision, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticDivisionFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), ArithmeticDivisionFP16Dataset),
                                                                                                                      EmptyActivationFunctionsDataset),
                                                                                                              InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, 0.01);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLArithmeticDivisionFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::TinyShapes(), ArithmeticDivisionFP16Dataset),
                                                                                                                       ActivationFunctionsDataset),
                                                                                                                       InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, 0.01);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticDivisionFloatFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), ArithmeticDivisionFP32Dataset),
                                                                                                                     EmptyActivationFunctionsDataset),
                                                                                                                     InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLArithmeticDivisionFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::TinyShapes(), ArithmeticDivisionFP32Dataset),
                                                                                                                        ActivationFunctionsDataset),
                                                                                                                        InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticDivisionFloatFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), ArithmeticDivisionFP32Dataset),
                                                                                                                   EmptyActivationFunctionsDataset),
                                                                                                                   InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}

template <typename T>
using CLArithmeticDivisionBroadcastFloatFixture = ArithmeticDivisionBroadcastValidationFloatFixture<CLTensor, CLAccessor, CLArithmeticDivision, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLArithmeticDivisionBroadcastFloatFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapesBroadcast(),
                       ArithmeticDivisionFP32Dataset),
                       EmptyActivationFunctionsDataset),
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunWithActivationBroadcast, CLArithmeticDivisionBroadcastFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::TinyShapesBroadcast(),
                       ArithmeticDivisionFP32Dataset),
                       ActivationFunctionsDataset),
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast, CLArithmeticDivisionBroadcastFloatFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapesBroadcast(),
                       ArithmeticDivisionFP32Dataset),
                       EmptyActivationFunctionsDataset),
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
