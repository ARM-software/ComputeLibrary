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
#include "arm_compute/core/Rounding.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PixelWiseMultiplicationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const float scale_unity = 1.f;
const float scale_255   = 1.f / 255.f;
const float scale_other = 1.f / 32768.f;

constexpr AbsoluteTolerance<float> tolerance_qasymm8(1); /**< Tolerance value for comparing reference's output against implementation's output for 8-bit quantized asymmetric data types */
constexpr AbsoluteTolerance<float> tolerance_qsymm16(1); /**< Tolerance value for comparing reference's output against implementation's output for 16-bit quantized symmetric data types */

const auto PixelWiseMultiplicationQSYMM16QuantDataset = combine(combine(
                                                                    framework::dataset::make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0) }),
                                                                    framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0) })),
                                                                framework::dataset::make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) }));

const auto PixelWiseMultiplicationQASYMM8QuantDataset = combine(combine(
                                                                    framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 32768.f, 0) }),
                                                                    framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0) })),
                                                                framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 32768.f, 0) }));

const auto PixelWiseMultiplicationPolicySTNUDataset = combine(
                                                          framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE }),
                                                          framework::dataset::make("RoundingPolicy", { RoundingPolicy::TO_NEAREST_UP }));

const auto PixelWiseMultiplicationPolicySTZDataset = combine(
                                                         framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE }),
                                                         framework::dataset::make("RoundingPolicy", { RoundingPolicy::TO_ZERO }));

/** Tests for in-place computation
 * With current interface storing TensorInfo with quantization information
 * in the kernel, it is difficult to have different tensor metadata
 * (e.g., quantization information, data type, different shape for broadcasting)
 * when an input is used as the output of the computation.
 * So, the following dataset for in-place computation is used only when
 * the exact same input and output Tensor object makes sense
 * (i.e., all the tensor metadata is the same) whereas if output is
 * expected to have either different quantization information, data type
 * or different shape we are not testing in-place computation.
 */
const auto InPlaceDataSet = framework::dataset::make("InPlace", { false, true });

#define DEFAULT_VALIDATE validate(Accessor(_target), _reference);
#define VALIDATE(TYPE, TOLERANCE) validate(Accessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);
#define WRAP_VALIDATE(TYPE, TOLERANCE) validate_wrap(Accessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);

// *INDENT-OFF*
// clang-format off
#define PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, DT3, SCALE, RP, INPLACE_DATASET, VALIDATE) \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, NEPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                        \
                           combine(combine(combine(combine(combine(combine(combine(                                            \
                           datasets::SHAPES,                                                                              \
                           framework::dataset::make("DataType1", DataType::DT1)),                                         \
                           framework::dataset::make("DataType2", DataType::DT2)),                                         \
                           framework::dataset::make("DataType3", DataType::DT3)),                                         \
                           framework::dataset::make("Scale", std::move(SCALE))),                                          \
                           datasets::ConvertPolicies()),                                                                  \
                           framework::dataset::make("RoundingPolicy", RoundingPolicy::RP)),                               \
                           (INPLACE_DATASET)))                                                                            \
    {                                                                                                                     \
        VALIDATE                                                                                                          \
    }

// *INDENT-ON*
// clang-format on
} // namespace

using NEPixelWiseMultiplicationQASYMM8Fixture       = PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, uint8_t, uint8_t>;
using NEPixelWiseMultiplicationQASYMM8SignedFixture = PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, int8_t, int8_t>;
using NEPixelWiseMultiplicationQSYMM16Fixture       = PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, int16_t, int16_t>;
using NEPixelWiseMultiplicationQSYMM16ToS32Fixture  = PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, int16_t, int16_t, int32_t>;
template <typename T>
using NEPixelWiseMultiplicationToU8Fixture = PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, uint8_t>;
template <typename T>
using NEPixelWiseMultiplicationToS16Fixture = PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, int16_t>;
template <typename T>
using NEPixelWiseMultiplicationToF16Fixture = PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, half_float::half>;
template <typename T>
using NEPixelWiseMultiplicationToF32Fixture = PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, float>;
template <typename T>
using NEPixelWiseMultiplicationBroadcastFixture = PixelWiseMultiplicationBroadcastValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, float>;
using NEPixelWiseMultiplicationU8U8ToS16Fixture = PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, uint8_t, uint8_t, int16_t>;

TEST_SUITE(NEON)
TEST_SUITE(PixelWiseMultiplication)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),                 //1 Ok
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),                 //2 Ok
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),                 //3 Window shrink
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),                 //4 Invalid scale
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),                 //5 Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),                //6 Mismatching shapes
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),                //7 Mismatching data type
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),            //8 Mismatching data type
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),     //9 Ok
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),     //10 Mismatching data type
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),            //11 Mismatching data type
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),            //12 Ok
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),     //13 Quantized cannot do WRAP
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                     })),
               framework::dataset::make("Scale",{  scale_unity,
                                                   scale_unity,
                                                   scale_unity,
                                                   -1.f,
                                                   scale_unity,
                                                   scale_unity,
                                                   scale_unity,
                                                   scale_unity,
                                                   scale_unity,
                                                   scale_unity,
                                                   scale_unity,
                                                   scale_unity,
                                                   scale_unity})),
               framework::dataset::make("OverflowPolicy",{
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::SATURATE,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::SATURATE,
                                                   ConvertPolicy::WRAP,
                                        })),

               framework::dataset::make("Expected", { true, true, true, false, false, false, false, false, true , false, false, true, false })),
               input1_info, input2_info, output_info, scale, policy, expected)
{
    bool has_error = bool(NEPixelWiseMultiplication::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), scale, policy, RoundingPolicy::TO_ZERO));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(InPlaceValidate)
TEST_CASE(SingleTensor, framework::DatasetMode::ALL)
{
    const auto random_shape       = TensorShape{ 9, 9 };
    const auto single_tensor_info = TensorInfo{ random_shape, 1, DataType::F32 };

    Status result = NEPixelWiseMultiplication::validate(&single_tensor_info, &single_tensor_info, &single_tensor_info, scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(ValidBroadCast, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{ 27U, 13U, 2U };
    const auto smaller_shape = TensorShape{ 1U, 13U, 2U };

    const auto larger_tensor_info  = TensorInfo{ larger_shape, 1, DataType::F32 };
    const auto smaller_tensor_info = TensorInfo{ smaller_shape, 1, DataType::F32 };

    Status result = NEPixelWiseMultiplication::validate(&larger_tensor_info, &smaller_tensor_info, &larger_tensor_info, scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastOutput, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{ 27U, 13U, 2U };
    const auto smaller_shape = TensorShape{ 1U, 13U, 2U };

    const auto larger_tensor_info  = TensorInfo{ larger_shape, 1, DataType::F32 };
    const auto smaller_tensor_info = TensorInfo{ smaller_shape, 1, DataType::F32 };

    Status result = NEPixelWiseMultiplication::validate(&larger_tensor_info, &smaller_tensor_info, &smaller_tensor_info, scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastBoth, framework::DatasetMode::ALL)
{
    const auto shape0 = TensorShape{ 9U, 9U };
    const auto shape1 = TensorShape{ 9U, 1U, 2U };

    const auto info0 = TensorInfo{ shape0, 1, DataType::F32 };
    const auto info1 = TensorInfo{ shape1, 1, DataType::F32 };

    Status result{};

    result = NEPixelWiseMultiplication::validate(&info0, &info1, &info0, scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEPixelWiseMultiplication::validate(&info0, &info1, &info1, scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // InPlaceValidate

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
TEST_SUITE(Scale255)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQASYMM8SignedFixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataTypeIn1", DataType::QASYMM8_SIGNED)),
                                                                                                                     framework::dataset::make("DataTypeIn2", DataType::QASYMM8_SIGNED)),
                                                                                                                     framework::dataset::make("DataTypeOut", DataType::QASYMM8_SIGNED)),
                                                                                                                     framework::dataset::make("Scale", { scale_unity })),
                                                                                                                     PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                     PixelWiseMultiplicationQASYMM8QuantDataset),
                                                                                                                     InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Scale255
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8)
TEST_SUITE(Scale255)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataTypeIn2", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataTypeOut", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("Scale", { scale_255 })),
                                                                                                                       PixelWiseMultiplicationPolicySTNUDataset),
                                                                                                                       PixelWiseMultiplicationQASYMM8QuantDataset),
                                                                                                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Scale255
TEST_SUITE(ScaleUnity)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataTypeIn2", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataTypeOut", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("Scale", { scale_unity })),
                                                                                                                       PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                       PixelWiseMultiplicationQASYMM8QuantDataset),
                                                                                                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // ScaleUnity
TEST_SUITE(ScaleOther)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataTypeIn2", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataTypeOut", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("Scale", { scale_other })),
                                                                                                                       PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                       PixelWiseMultiplicationQASYMM8QuantDataset),
                                                                                                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // ScaleOther
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QSYMM16)
TEST_SUITE(Scale255)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("DataTypeIn2", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("DataTypeOut", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("Scale", { scale_255 })),
                                                                                                                       PixelWiseMultiplicationPolicySTNUDataset),
                                                                                                                       PixelWiseMultiplicationQSYMM16QuantDataset),
                                                                                                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // Scale255
TEST_SUITE(ScaleUnity)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("DataTypeIn2", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("DataTypeOut", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("Scale", { scale_unity })),
                                                                                                                       PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                       PixelWiseMultiplicationQSYMM16QuantDataset),
                                                                                                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // ScaleUnity
TEST_SUITE(ScaleOther)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("DataTypeIn2", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("DataTypeOut", DataType::QSYMM16)),
                                                                                                                       framework::dataset::make("Scale", { scale_other })),
                                                                                                                       PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                       PixelWiseMultiplicationQSYMM16QuantDataset),
                                                                                                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // ScaleOther
TEST_SUITE_END() // QSYMM16
TEST_SUITE(QSYMM16toS32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQSYMM16ToS32Fixture, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                    framework::dataset::make("DataTypeIn1", DataType::QSYMM16)),
                                                                                                                    framework::dataset::make("DataTypeIn2", DataType::QSYMM16)),
                                                                                                                    framework::dataset::make("DataTypeOut", DataType::S32)),
                                                                                                                    framework::dataset::make("Scale", { scale_unity })),
                                                                                                                    PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                    PixelWiseMultiplicationQSYMM16QuantDataset),
                                                                                                                    framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QSYMM16toS32
TEST_SUITE_END() // Quantized

TEST_SUITE(U8U8toS16)

FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationU8U8ToS16Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataTypeIn1", DataType::U8)),
                                                                                                                       framework::dataset::make("DataTypeIn2", DataType::U8)),
                                                                                                                       framework::dataset::make("DataTypeOut", DataType::S16)),
                                                                                                                       framework::dataset::make("Scale", { scale_255 })),
                                                                                                                       datasets::ConvertPolicies()),
                                                                                                                       framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_NEAREST_UP)),
                                                                                                                       framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate_wrap(Accessor(_target), _reference, AbsoluteTolerance<int16_t>(1), 0.f);
}

FIXTURE_DATA_TEST_CASE(RunSmall1, NEPixelWiseMultiplicationU8U8ToS16Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                        framework::dataset::make("DataTypeIn1", DataType::U8)),
                                                                                                                        framework::dataset::make("DataTypeIn2", DataType::U8)),
                                                                                                                        framework::dataset::make("DataTypeOut", DataType::S16)),
                                                                                                                        framework::dataset::make("Scale", { scale_other })),
                                                                                                                        datasets::ConvertPolicies()),
                                                                                                                        framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_ZERO)),
                                                                                                                        framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END() // U8U8toS16

TEST_SUITE(U8toU8)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToU8Fixture<uint8_t>, ALL, SmallShapes(), U8, U8, U8, scale_255, TO_NEAREST_UP, InPlaceDataSet, WRAP_VALIDATE(uint8_t, 1))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToU8Fixture<uint8_t>, ALL, SmallShapes(), U8, U8, U8, scale_unity, TO_ZERO, InPlaceDataSet, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToU8Fixture<uint8_t>, ALL, SmallShapes(), U8, U8, U8, scale_other, TO_ZERO, InPlaceDataSet, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // U8toU8

TEST_SUITE(U8toS16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<uint8_t>, ALL, SmallShapes(), U8, S16, S16, scale_255, TO_NEAREST_UP, framework::dataset::make("InPlace", { false }),
                                                 WRAP_VALIDATE(int16_t, 2))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<uint8_t>, ALL, SmallShapes(), U8, S16, S16, scale_unity, TO_ZERO, framework::dataset::make("InPlace", { false }),
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<uint8_t>, ALL, SmallShapes(), U8, S16, S16, scale_other, TO_ZERO, framework::dataset::make("InPlace", { false }),
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // U8toS16

TEST_SUITE(S16toS16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<int16_t>, ALL, SmallShapes(), S16, S16, S16, scale_255, TO_NEAREST_UP, InPlaceDataSet, WRAP_VALIDATE(int16_t, 2))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<int16_t>, ALL, SmallShapes(), S16, S16, S16, scale_unity, TO_ZERO, InPlaceDataSet, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<int16_t>, ALL, SmallShapes(), S16, S16, S16, scale_other, TO_ZERO, InPlaceDataSet, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // S16toS16

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16toF16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF16Fixture<half_float::half>, ALL, SmallShapes(), F16, F16, F16, scale_255, TO_NEAREST_UP, InPlaceDataSet, VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE_END() // F16toF16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32toF32)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, ALL, SmallShapes(), F32, F32, F32, scale_255, TO_NEAREST_UP, InPlaceDataSet, VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, ALL, SmallShapes(), F32, F32, F32, scale_unity, TO_ZERO, InPlaceDataSet, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, ALL, SmallShapes(), F32, F32, F32, scale_other, TO_ZERO, InPlaceDataSet, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // F32toF32

TEST_SUITE(Broadcast)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, BroadcastFixture<float>, ALL, SmallShapesBroadcast(), F32, F32, F32, scale_255, TO_NEAREST_UP, framework::dataset::make("InPlace", { false }),
                                                 VALIDATE(float, 1.f))
TEST_SUITE_END() // Broadcast

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
