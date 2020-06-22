/*
 * Copyright (c) 2017-2020 ARM Limited.
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
/** Input data sets **/
const auto ArithmeticAdditionU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)), framework::dataset::make("DataType",
                                                 DataType::U8));
const auto ArithmeticAdditionQASYMM8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::QASYMM8)),
                                                      framework::dataset::make("DataType",
                                                                               DataType::QASYMM8));
const auto ArithmeticAdditionQASYMM8SignedDataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8_SIGNED), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                            framework::dataset::make("DataType",
                                                                                     DataType::QASYMM8_SIGNED));
const auto ArithmeticAdditionQSYMM16Dataset = combine(combine(framework::dataset::make("DataType", DataType::QSYMM16), framework::dataset::make("DataType", DataType::QSYMM16)),
                                                      framework::dataset::make("DataType",
                                                                               DataType::QSYMM16));
const auto ArithmeticAdditionS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::U8, DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                                  framework::dataset::make("DataType", DataType::S16));
const auto ArithmeticAdditionFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                                   framework::dataset::make("DataType", DataType::F16));
const auto ArithmeticAdditionFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                                   framework::dataset::make("DataType", DataType::F32));
const auto EmptyActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo() });
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.75f, 0.25f)
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ArithmeticAddition)

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
    ARM_COMPUTE_EXPECT(bool(CLArithmeticAddition::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), ConvertPolicy::WRAP)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLArithmeticAdditionFixture = ArithmeticAdditionValidationFixture<CLTensor, CLAccessor, CLArithmeticAddition, T>;

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticAdditionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticAdditionU8Dataset),
                                                                                                                  framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticAdditionFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticAdditionS16Dataset),
                                                                                                                  framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticAdditionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ArithmeticAdditionS16Dataset),
                                                                                                                framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Integer

template <typename T>
using CLArithmeticAdditionQuantizedFixture = ArithmeticAdditionValidationQuantizedFixture<CLTensor, CLAccessor, CLArithmeticAddition, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticAdditionQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallShapes(),
                       ArithmeticAdditionQASYMM8Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                       framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticAdditionQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallShapes(),
                       ArithmeticAdditionQASYMM8SignedDataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 10) })),
                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                       framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticAdditionQuantizedFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallShapes(),
                       ArithmeticAdditionQSYMM16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       framework::dataset::make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                       framework::dataset::make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

template <typename T>
using CLArithmeticAdditionFloatFixture = ArithmeticAdditionValidationFloatFixture<CLTensor, CLAccessor, CLArithmeticAddition, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticAdditionFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), ArithmeticAdditionFP16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                              EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLArithmeticAdditionFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::TinyShapes(), ArithmeticAdditionFP16Dataset),
                                                                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                       ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticAdditionFloatFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), ArithmeticAdditionFP32Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLArithmeticAdditionFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::TinyShapes(), ArithmeticAdditionFP32Dataset),
                                                                                                                        framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                        ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticAdditionFloatFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), ArithmeticAdditionFP32Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

template <typename T>
using CLArithmeticAdditionBroadcastFloatFixture = ArithmeticAdditionBroadcastValidationFloatFixture<CLTensor, CLAccessor, CLArithmeticAddition, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLArithmeticAdditionBroadcastFloatFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapesBroadcast(),
                       ArithmeticAdditionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunWithActivationBroadcast, CLArithmeticAdditionBroadcastFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::TinyShapesBroadcast(),
                       ArithmeticAdditionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast, CLArithmeticAdditionBroadcastFloatFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapesBroadcast(),
                       ArithmeticAdditionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ArithmeticAddition
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
