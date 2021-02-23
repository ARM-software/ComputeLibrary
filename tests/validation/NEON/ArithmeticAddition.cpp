/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
#if !defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
constexpr AbsoluteTolerance<float> tolerance_quant(1); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
#else                                                  // !defined(__aarch64__) || defined(__ARM_FEATURE_SVE)
constexpr AbsoluteTolerance<float> tolerance_quant(0);
#endif                                                 // !defined(__aarch64__) || defined(__ARM_FEATURE_SVE)

/** Input data sets **/
const auto ArithmeticAdditionU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)), framework::dataset::make("DataType",
                                                 DataType::U8));
const auto ArithmeticAdditionS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::U8, DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                                  framework::dataset::make("DataType", DataType::S16));
const auto ArithmeticAdditionS32Dataset = combine(combine(framework::dataset::make("DataType", { DataType::S32 }), framework::dataset::make("DataType", DataType::S32)),
                                                  framework::dataset::make("DataType", DataType::S32));
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const auto ArithmeticAdditionFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                                   framework::dataset::make("DataType", DataType::F16));
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const auto ArithmeticAdditionFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                                   framework::dataset::make("DataType", DataType::F32));
const auto ArithmeticAdditionQASYMM8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::QASYMM8)),
                                                      framework::dataset::make("DataType", DataType::QASYMM8));
const auto ArithmeticAdditionQASYMM8SIGNEDDataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8_SIGNED), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                            framework::dataset::make("DataType", DataType::QASYMM8_SIGNED));
const auto ArithmeticAdditionQSYMM16Dataset = combine(combine(framework::dataset::make("DataType", DataType::QSYMM16), framework::dataset::make("DataType", DataType::QSYMM16)),
                                                      framework::dataset::make("DataType", DataType::QSYMM16));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ArithmeticAddition)

template <typename T>
using NEArithmeticAdditionFixture = ArithmeticAdditionValidationFixture<Tensor, Accessor, NEArithmeticAddition, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8), // Unsupported broadcast
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),// Mismatching shapes
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(1U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, true, false, false, false})),
               input1_info, input2_info, output_info, expected)
{
    Status s = NEArithmeticAddition::validate(&input1_info.clone()->set_is_resizable(false),
                                              &input2_info.clone()->set_is_resizable(false),
                                              &output_info.clone()->set_is_resizable(false),
                                              ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_CASE(NoPaddingAdded, framework::DatasetMode::PRECOMMIT)
{
    // NEArithmeticAddition doesn't use padding, so make sure this is the case.
    Tensor input1 = create_tensor<Tensor>(TensorShape(15U, 15U), DataType::F32);
    Tensor input2 = create_tensor<Tensor>(TensorShape(15U, 1U), DataType::F32);
    Tensor output = create_tensor<Tensor>(TensorShape(15U, 15U), DataType::F32);

    NEArithmeticAddition add;
    add.configure(&input1, &input2, &output, ConvertPolicy::WRAP);

    // Validate padding is zero
    validate(input1.info()->padding(), PaddingSize());
    validate(input2.info()->padding(), PaddingSize());
    validate(output.info()->padding(), PaddingSize());
}

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticAdditionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticAdditionU8Dataset),
                                                                                                                  framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticAdditionFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticAdditionS16Dataset),
                                                                                                                  framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEArithmeticAdditionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ArithmeticAdditionS16Dataset),
                                                                                                                framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticAdditionFixture<int32_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ArithmeticAdditionS32Dataset),
                                                                                                            framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32
TEST_SUITE_END() // Integer

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticAdditionFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ArithmeticAdditionFP16Dataset),
                                                                                                         framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticAdditionFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticAdditionFP32Dataset),
                                                                                                                framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEArithmeticAdditionFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ArithmeticAdditionFP32Dataset),
                                                                                                              framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

template <typename T>
using NEArithmeticAdditionBroadcastFixture = ArithmeticAdditionBroadcastValidationFixture<Tensor, Accessor, NEArithmeticAddition, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NEArithmeticAdditionBroadcastFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapesBroadcast(),
                       ArithmeticAdditionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast, NEArithmeticAdditionBroadcastFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapesBroadcast(),
                       ArithmeticAdditionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

template <typename T>
using NEArithmeticAdditionQuantizedFixture = ArithmeticAdditionValidationQuantizedFixture<Tensor, Accessor, NEArithmeticAddition, T>;

template <typename T>
using NEArithmeticAdditionQuantizedBroadcastFixture = ArithmeticAdditionValidationQuantizedBroadcastFixture<Tensor, Accessor, NEArithmeticAddition, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticAdditionQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(datasets::SmallShapes(), ArithmeticAdditionQASYMM8Dataset),
                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                               framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                               framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticAdditionQuantizedFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::SmallShapes(), ArithmeticAdditionQASYMM8SIGNEDDataset),
                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                               framework::dataset::make("Src0QInfo", { QuantizationInfo(0.5f, 20) })),
                                       framework::dataset::make("Src1QInfo", { QuantizationInfo(0.5f, 10) })),
                               framework::dataset::make("OutQInfo", { QuantizationInfo(0.5f, 5) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quant);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NEArithmeticAdditionQuantizedBroadcastFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(
                           datasets::SmallShapesBroadcast(), ArithmeticAdditionQASYMM8SIGNEDDataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                       framework::dataset::make("Src0QInfo", { QuantizationInfo(0.5f, 20) })),
                       framework::dataset::make("Src1QInfo", { QuantizationInfo(0.5f, 10) })),
                       framework::dataset::make("OutQInfo", { QuantizationInfo(0.5f, 5) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticAdditionQuantizedFixture<int16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(datasets::SmallShapes(), ArithmeticAdditionQSYMM16Dataset),
                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                               framework::dataset::make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                               framework::dataset::make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // ArithmeticAddition
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
