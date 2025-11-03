/*
 * Copyright (c) 2018-2021, 2025 Arm Limited.
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
using framework::dataset::make;
namespace
{
RelativeTolerance<float> tolerance_fp32(0.000001f);
RelativeTolerance<float> tolerance_fp16(0.001f);

/** Input data sets **/
const auto ElementwiseMaxU8Dataset = combine(make("DataType", DataType::U8), make("DataType", DataType::U8), make("DataType",
                                             DataType::U8));
const auto ElementwiseMaxQASYMM8Dataset = combine(make("DataType", DataType::QASYMM8), make("DataType", DataType::QASYMM8),
                                                  make("DataType",
                                                                           DataType::QASYMM8));
const auto ElementwiseMaxQASYMM8SignedDataset = combine(make("DataType", DataType::QASYMM8_SIGNED), make("DataType", DataType::QASYMM8_SIGNED),
                                                        make("DataType",
                                                                                 DataType::QASYMM8_SIGNED));
const auto ElementwiseMaxQSYMM16Dataset = combine(make("DataType", DataType::QSYMM16), make("DataType", DataType::QSYMM16),
                                                  make("DataType",
                                                                           DataType::QSYMM16));
const auto ElementwiseMaxS16Dataset = combine(make("DataType", { DataType::S16 }), make("DataType", DataType::S16),
                                              make("DataType", DataType::S16));
const auto ElementwiseMaxFP16Dataset = combine(make("DataType", DataType::F16), make("DataType", DataType::F16),
                                               make("DataType", DataType::F16));
const auto ElementwiseMaxFP32Dataset = combine(make("DataType", DataType::F32), make("DataType", DataType::F32),
                                               make("DataType", DataType::F32));
const auto EmptyActivationFunctionsDataset = make("ActivationInfo",
{ ActivationLayerInfo() });
const auto ActivationFunctionsDataset = make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.75f, 0.25f)
});
const auto InPlaceDataSet    = make("InPlace", { false, true });
const auto OutOfPlaceDataSet = make("InPlace", { false });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ElementwiseMax)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     }),
               make("OutputInfo",{TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     }),
               make("Expected", { true, false, false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLElementwiseMax::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLElementwiseMaxFixture = ElementwiseMaxValidationFixture<CLTensor, CLAccessor, CLElementwiseMax, T>;

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMaxFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), ElementwiseMaxU8Dataset,
                                                                                                              OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMaxFixture<int16_t>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwiseMaxS16Dataset,
                                                                                                        OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLElementwiseMaxQuantizedFixture = ElementwiseMaxValidationQuantizedFixture<CLTensor, CLAccessor, CLElementwiseMax, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMaxQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(),
                                                                                                                       ElementwiseMaxQASYMM8Dataset,
                                                                                                                       make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) }),
                                                                                                                       make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) }),
                                                                                                                       make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) }),
                                                                                                                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END()
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMaxQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(),
                                                                                                                      ElementwiseMaxQASYMM8SignedDataset,
                                                                                                                      make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) }),
                                                                                                                      make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) }),
                                                                                                                      make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) }),
                                                                                                                      OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMaxQuantizedFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(),
                                                                                                                       ElementwiseMaxQSYMM16Dataset,
                                                                                                                       make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) }),
                                                                                                                       make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) }),
                                                                                                                       make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) }),
                                                                                                                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLElementwiseMaxFloatFixture = ElementwiseMaxValidationFloatFixture<CLTensor, CLAccessor, CLElementwiseMax, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMaxFloatFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwiseMaxFP16Dataset,
                                                                                                                  EmptyActivationFunctionsDataset,
                                                                                                          OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, 0.01);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLElementwiseMaxFloatFixture<half>, framework::DatasetMode::ALL, combine(datasets::TinyShapes(), ElementwiseMaxFP16Dataset,
                                                                                                                   ActivationFunctionsDataset,
                                                                                                                   OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, 0.01);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMaxFloatFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwiseMaxFP32Dataset,
                                                                                                                   EmptyActivationFunctionsDataset,
                                                                                                           OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLElementwiseMaxFloatFixture<float>, framework::DatasetMode::ALL, combine(datasets::TinyShapes(), ElementwiseMaxFP32Dataset,
                                                                                                                    ActivationFunctionsDataset,
                                                                                                                    OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}

template <typename T>
using CLElementwiseMaxBroadcastFloatFixture = ElementwiseMaxBroadcastValidationFloatFixture<CLTensor, CLAccessor, CLElementwiseMax, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLElementwiseMaxBroadcastFloatFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapesBroadcast(),
                       ElementwiseMaxFP32Dataset,
                       EmptyActivationFunctionsDataset,
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunWithActivationBroadcast, CLElementwiseMaxBroadcastFloatFixture<float>, framework::DatasetMode::ALL, combine(datasets::TinyShapesBroadcast(),
                       ElementwiseMaxFP32Dataset,
                       ActivationFunctionsDataset,
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END() // ElementwiseMax
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
