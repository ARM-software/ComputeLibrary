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
#include "tests/validation/fixtures/ArithmeticOperationsFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;
/** Input data sets **/
const auto ArithmeticSubtractionU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)),
                                                    framework::dataset::make("DataType",
                                                                             DataType::U8));
const auto ArithmeticSubtractionQASYMM8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::QASYMM8)),
                                                         framework::dataset::make("DataType",
                                                                                  DataType::QASYMM8));
const auto ArithmeticSubtractionQASYMM8SignedDataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8_SIGNED), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                               framework::dataset::make("DataType",
                                                                                        DataType::QASYMM8_SIGNED));
const auto ArithmeticSubtractionQSYMM16Dataset = combine(combine(framework::dataset::make("DataType", DataType::QSYMM16), framework::dataset::make("DataType", DataType::QSYMM16)),
                                                         framework::dataset::make("DataType",
                                                                                  DataType::QSYMM16));
const auto ArithmeticSubtractionS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::U8, DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                                     framework::dataset::make("DataType", DataType::S16));
const auto ArithmeticSubtractionFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                                      framework::dataset::make("DataType", DataType::F16));
const auto ArithmeticSubtractionFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                                      framework::dataset::make("DataType", DataType::F32));
const auto EmptyActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo() });
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.75f, 0.25f)
});
const auto InPlaceDataSet    = framework::dataset::make("InPlace", { false, true });
const auto OutOfPlaceDataSet = framework::dataset::make("InPlace", { false });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ArithmeticSubtraction)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),      // Window shrink
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, true, false, false, false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLArithmeticSubtraction::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), ConvertPolicy::WRAP)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(InPlaceValidate)
TEST_CASE(SingleTensor, framework::DatasetMode::ALL)
{
    const auto random_shape       = TensorShape{ 9, 9 };
    const auto single_tensor_info = TensorInfo{ random_shape, 1, DataType::F32 };

    Status result = CLArithmeticSubtraction::validate(&single_tensor_info, &single_tensor_info, &single_tensor_info, ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(ValidBroadCast, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{ 27U, 13U, 2U };
    const auto smaller_shape = TensorShape{ 1U, 13U, 2U };

    const auto larger_tensor_info  = TensorInfo{ larger_shape, 1, DataType::F32 };
    const auto smaller_tensor_info = TensorInfo{ smaller_shape, 1, DataType::F32 };

    Status result = CLArithmeticSubtraction::validate(&larger_tensor_info, &smaller_tensor_info, &larger_tensor_info, ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastOutput, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{ 27U, 13U, 2U };
    const auto smaller_shape = TensorShape{ 1U, 13U, 2U };

    const auto larger_tensor_info  = TensorInfo{ larger_shape, 1, DataType::F32 };
    const auto smaller_tensor_info = TensorInfo{ smaller_shape, 1, DataType::F32 };

    Status result = CLArithmeticSubtraction::validate(&larger_tensor_info, &smaller_tensor_info, &smaller_tensor_info, ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastBoth, framework::DatasetMode::ALL)
{
    const auto shape0 = TensorShape{ 9U, 9U };
    const auto shape1 = TensorShape{ 9U, 1U, 2U };

    const auto info0 = TensorInfo{ shape0, 1, DataType::F32 };
    const auto info1 = TensorInfo{ shape1, 1, DataType::F32 };

    Status result{};

    result = CLArithmeticSubtraction::validate(&info0, &info1, &info0, ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = CLArithmeticSubtraction::validate(&info0, &info1, &info1, ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // InPlaceValidate

template <typename T>
using CLArithmeticSubtractionFixture = ArithmeticSubtractionValidationFixture<CLTensor, CLAccessor, CLArithmeticSubtraction, T>;

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), ArithmeticSubtractionU8Dataset),
                                                                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                               OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), ArithmeticSubtractionS16Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticSubtractionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), ArithmeticSubtractionS16Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Integer

template <typename T>
using CLArithmeticSubtractionQuantizedFixture = ArithmeticSubtractionValidationQuantizedFixture<CLTensor, CLAccessor, CLArithmeticSubtraction, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                       ArithmeticSubtractionQASYMM8Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                       framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) })),
                       InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                       ArithmeticSubtractionQASYMM8SignedDataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 10) })),
                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                       framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) })),
                       InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionQuantizedFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                       ArithmeticSubtractionQSYMM16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       framework::dataset::make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                       framework::dataset::make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) })),
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

template <typename T>
using CLArithmeticSubtractionFloatFixture = ArithmeticSubtractionValidationFloatFixture<CLTensor, CLAccessor, CLArithmeticSubtraction, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallShapes(), ArithmeticSubtractionFP16Dataset),
                                                                                                                 framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                 EmptyActivationFunctionsDataset),
                                                                                                                 OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLArithmeticSubtractionFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::TinyShapes(),
                       ArithmeticSubtractionFP16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       ActivationFunctionsDataset),
                       InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFloatFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                        ArithmeticSubtractionFP32Dataset),
                                                                                                                        framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                        EmptyActivationFunctionsDataset),
                                                                                                                        OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLArithmeticSubtractionFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::TinyShapes(),
                       ArithmeticSubtractionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       ActivationFunctionsDataset),
                       InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticSubtractionFloatFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                      ArithmeticSubtractionFP32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      EmptyActivationFunctionsDataset),
                                                                                                                      OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

template <typename T>
using CLArithmeticSubtractionBroadcastFloatFixture = ArithmeticSubtractionBroadcastValidationFloatFixture<CLTensor, CLAccessor, CLArithmeticSubtraction, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLArithmeticSubtractionBroadcastFloatFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapesBroadcast(),
                       ArithmeticSubtractionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       EmptyActivationFunctionsDataset),
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunWithActivationBroadcast, CLArithmeticSubtractionBroadcastFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::TinyShapesBroadcast(),
                       ArithmeticSubtractionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       ActivationFunctionsDataset),
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast, CLArithmeticSubtractionBroadcastFloatFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapesBroadcast(),
                       ArithmeticSubtractionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       EmptyActivationFunctionsDataset),
                       OutOfPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ArithmeticSubtraction
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
