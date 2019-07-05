/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#define DEFAULT_VALIDATE validate(Accessor(_target), _reference);
#define VALIDATE(TYPE, TOLERANCE) validate(Accessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);
#define WRAP_VALIDATE(TYPE, TOLERANCE) validate_wrap(Accessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);

// *INDENT-OFF*
// clang-format off
#define PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(DT1, DT2, SCALE, RP)                                            \
    DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL,                                                   \
                   combine(combine(combine(combine(combine(                                                      \
                   concat(datasets::SmallShapes(), datasets::LargeShapes()),                                     \
                   framework::dataset::make("DataType1", DataType::DT1)),                                        \
                   framework::dataset::make("DataType2", DataType::DT2)),                                        \
                   framework::dataset::make("Scale", std::move(SCALE))),                                         \
                   datasets::ConvertPolicies()),                                                                 \
                   framework::dataset::make("RoundingPolicy", RoundingPolicy::RP)),                              \
                   shape, dt1, dt2, scale, convert_policy, rounding_policy)                                      \
    {                                                                                                            \
        validate_configuration(shape, dt1, dt2, scale, convert_policy, rounding_policy);                         \
    }

#define PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, SCALE, RP, VALIDATE) \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, NEPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                   \
                           combine(combine(combine(combine(combine(                                                       \
                           datasets::SHAPES,                                                                              \
                           framework::dataset::make("DataType1", DataType::DT1)),                                         \
                           framework::dataset::make("DataType2", DataType::DT2)),                                         \
                           framework::dataset::make("Scale", std::move(SCALE))),                                          \
                           datasets::ConvertPolicies()),                                                                  \
                           framework::dataset::make("RoundingPolicy", RoundingPolicy::RP)))                               \
    {                                                                                                                     \
        VALIDATE                                                                                                          \
    }

// *INDENT-ON*
// clang-format on

void validate_configuration(TensorShape shape, DataType dt1, DataType dt2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    Tensor src1 = create_tensor<Tensor>(shape, dt1);
    Tensor src2 = create_tensor<Tensor>(shape, dt2);
    Tensor dst  = create_tensor<Tensor>(shape, dt2);

    ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src2.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEPixelWiseMultiplication multiply;
    multiply.configure(&src1, &src2, &dst, scale, convert_policy, rounding_policy);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src1.info()->valid_region(), valid_region);
    validate(src2.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src1.info()->padding(), padding);
    validate(src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}
} // namespace

using NEPixelWiseMultiplicationQASYMM8Fixture = PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, uint8_t, uint8_t>;
using NEPixelWiseMultiplicationQSYMM16Fixture = PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, int16_t, int16_t>;
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

TEST_SUITE(NEON)
TEST_SUITE(PixelWiseMultiplication)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),      // Window shrink
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid scale
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching data type
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8), // Mismatching data type
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                     })),
               framework::dataset::make("Scale",{  scale_unity, scale_unity, scale_unity, -1.f, scale_unity, scale_unity, scale_unity})),
               framework::dataset::make("Expected", { true, true, false, false, false, false, false, false })),
               input1_info, input2_info, output_info, scale, expected)
{
    bool has_error = bool(NEPixelWiseMultiplication::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), scale, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
TEST_SUITE(Scale255)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                     framework::dataset::make("Scale", { scale_255 })),
                                                                                                                     PixelWiseMultiplicationPolicySTNUDataset),
                                                                                                                     PixelWiseMultiplicationQASYMM8QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                   framework::dataset::make("Scale", { scale_255 })),
                                                                                                                   PixelWiseMultiplicationPolicySTNUDataset),
                                                                                                                   PixelWiseMultiplicationQASYMM8QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Scale255
TEST_SUITE(ScaleUnity)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                     framework::dataset::make("Scale", { scale_unity })),
                                                                                                                     PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                     PixelWiseMultiplicationQASYMM8QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                   framework::dataset::make("Scale", { scale_unity })),
                                                                                                                   PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                   PixelWiseMultiplicationQASYMM8QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // ScaleUnity
TEST_SUITE(ScaleOther)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                     framework::dataset::make("Scale", { scale_other })),
                                                                                                                     PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                     PixelWiseMultiplicationQASYMM8QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPixelWiseMultiplicationQASYMM8Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                   framework::dataset::make("Scale", { scale_other })),
                                                                                                                   PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                   PixelWiseMultiplicationQASYMM8QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // ScaleOther
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QSYMM16)
TEST_SUITE(Scale255)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QSYMM16)),
                                                                                                                     framework::dataset::make("Scale", { scale_255 })),
                                                                                                                     PixelWiseMultiplicationPolicySTNUDataset),
                                                                                                                     PixelWiseMultiplicationQSYMM16QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QSYMM16)),
                                                                                                                   framework::dataset::make("Scale", { scale_255 })),
                                                                                                                   PixelWiseMultiplicationPolicySTNUDataset),
                                                                                                                   PixelWiseMultiplicationQSYMM16QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // Scale255
TEST_SUITE(ScaleUnity)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QSYMM16)),
                                                                                                                     framework::dataset::make("Scale", { scale_unity })),
                                                                                                                     PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                     PixelWiseMultiplicationQSYMM16QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QSYMM16)),
                                                                                                                   framework::dataset::make("Scale", { scale_unity })),
                                                                                                                   PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                   PixelWiseMultiplicationQSYMM16QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // ScaleUnity
TEST_SUITE(ScaleOther)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QSYMM16)),
                                                                                                                     framework::dataset::make("Scale", { scale_other })),
                                                                                                                     PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                     PixelWiseMultiplicationQSYMM16QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPixelWiseMultiplicationQSYMM16Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QSYMM16)),
                                                                                                                   framework::dataset::make("Scale", { scale_other })),
                                                                                                                   PixelWiseMultiplicationPolicySTZDataset),
                                                                                                                   PixelWiseMultiplicationQSYMM16QuantDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // ScaleOther
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

TEST_SUITE(U8toU8)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(U8, U8, scale_255, TO_NEAREST_UP)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToU8Fixture<uint8_t>, PRECOMMIT, SmallShapes(), U8, U8, scale_255, TO_NEAREST_UP, WRAP_VALIDATE(uint8_t, 1))
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToU8Fixture<uint8_t>, NIGHTLY, LargeShapes(), U8, U8, scale_255, TO_NEAREST_UP, WRAP_VALIDATE(uint8_t, 1))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(U8, U8, scale_unity, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToU8Fixture<uint8_t>, PRECOMMIT, SmallShapes(), U8, U8, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToU8Fixture<uint8_t>, NIGHTLY, LargeShapes(), U8, U8, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(U8, U8, scale_other, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToU8Fixture<uint8_t>, PRECOMMIT, SmallShapes(), U8, U8, scale_other, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToU8Fixture<uint8_t>, NIGHTLY, LargeShapes(), U8, U8, scale_other, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // U8toU8

TEST_SUITE(U8toS16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(U8, S16, scale_255, TO_NEAREST_UP)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<uint8_t>, PRECOMMIT, SmallShapes(), U8, S16, scale_255, TO_NEAREST_UP, WRAP_VALIDATE(int16_t, 2))
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToS16Fixture<uint8_t>, NIGHTLY, LargeShapes(), U8, S16, scale_255, TO_NEAREST_UP, WRAP_VALIDATE(int16_t, 2))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(U8, S16, scale_unity, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<uint8_t>, PRECOMMIT, SmallShapes(), U8, S16, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToS16Fixture<uint8_t>, NIGHTLY, LargeShapes(), U8, S16, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(U8, S16, scale_other, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<uint8_t>, PRECOMMIT, SmallShapes(), U8, S16, scale_other, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToS16Fixture<uint8_t>, NIGHTLY, LargeShapes(), U8, S16, scale_other, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // U8toS16

TEST_SUITE(S16toS16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(S16, S16, scale_255, TO_NEAREST_UP)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<int16_t>, PRECOMMIT, SmallShapes(), S16, S16, scale_255, TO_NEAREST_UP, WRAP_VALIDATE(int16_t, 2))
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToS16Fixture<int16_t>, NIGHTLY, LargeShapes(), S16, S16, scale_255, TO_NEAREST_UP, WRAP_VALIDATE(int16_t, 2))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(S16, S16, scale_unity, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<int16_t>, PRECOMMIT, SmallShapes(), S16, S16, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToS16Fixture<int16_t>, NIGHTLY, LargeShapes(), S16, S16, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(S16, S16, scale_other, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToS16Fixture<int16_t>, PRECOMMIT, SmallShapes(), S16, S16, scale_other, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToS16Fixture<int16_t>, NIGHTLY, LargeShapes(), S16, S16, scale_other, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // S16toS16

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16toF16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF16Fixture<half_float::half>, PRECOMMIT, SmallShapes(), F16, F16, scale_255, TO_NEAREST_UP, VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE_END() // F16toF16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32toF32)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(F32, F32, scale_255, TO_NEAREST_UP)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, PRECOMMIT, SmallShapes(), F32, F32, scale_255, TO_NEAREST_UP, VALIDATE(float, 1.f))
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToF32Fixture<float>, NIGHTLY, LargeShapes(), F32, F32, scale_255, TO_NEAREST_UP, VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(F32, F32, scale_unity, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, PRECOMMIT, SmallShapes(), F32, F32, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToF32Fixture<float>, NIGHTLY, LargeShapes(), F32, F32, scale_unity, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_DATA_TEST_CASE(F32, F32, scale_other, TO_ZERO)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, PRECOMMIT, SmallShapes(), F32, F32, scale_other, TO_ZERO, DEFAULT_VALIDATE)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, ToF32Fixture<float>, NIGHTLY, LargeShapes(), F32, F32, scale_other, TO_ZERO, DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // F32toF32

TEST_SUITE(Broadcast)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, BroadcastFixture<float>, PRECOMMIT, SmallShapesBroadcast(), F32, F32, scale_255, TO_NEAREST_UP, VALIDATE(float, 1.f))
TEST_SUITE_END() // Broadcast

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
