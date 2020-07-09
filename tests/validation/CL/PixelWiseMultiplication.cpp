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
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"
#include "tests/CL/CLAccessor.h"
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
namespace
{
const float                        scale_255 = 1.f / 255.f;
constexpr AbsoluteTolerance<float> tolerance_qasymm8(1); /**< Tolerance value for comparing reference's output against implementation's output for 8-bit quantized asymmetric data types */
constexpr AbsoluteTolerance<float> tolerance_qsymm16(1); /**< Tolerance value for comparing reference's output against implementation's output for 16-bit quantized symmetric data types */
const auto                         EmptyActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo() });
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.75f, 0.25f)
});
// Since in-place computation on CL-side hasn't been intended to be implemented, they are not tested.
// However, this dataset is required for the shared fixture and it would make extension easier when
// CL-side also starts supporting in-place computation.
const auto InPlaceDataSet = framework::dataset::make("InPlace", { false });
} //namespace
// *INDENT-OFF*
// clang-format off
#define VALIDATE(TYPE, TOLERANCE) validate(CLAccessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);

#define PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, SCALE, RP, ACT, VALIDATE) \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, CLPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                   \
                           combine(combine(combine(combine(combine(combine(combine(                                                       \
                           datasets::SHAPES,                                                                              \
                           framework::dataset::make("DataType1", DataType::DT1)),                                         \
                           framework::dataset::make("DataType2", DataType::DT2)),                                         \
                           framework::dataset::make("Scale", std::move(SCALE))),                                          \
                           datasets::ConvertPolicies()),                                                                  \
                           framework::dataset::make("RoundingPolicy", RoundingPolicy::RP)), ACT), \
                           InPlaceDataSet))  \
    {                                                                                                                     \
        VALIDATE                                                                                                          \
    }
// clang-format on
// *INDENT-ON*
} // namespace

template <typename T>
using CLPixelWiseMultiplicationToF16Fixture = PixelWiseMultiplicationValidationFloatFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, half_float::half>;
template <typename T>
using CLPixelWiseMultiplicationToF32Fixture = PixelWiseMultiplicationValidationFloatFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, float>;
template <typename T>
using CLPixelWiseMultiplicationBroadcastFixture = PixelWiseMultiplicationBroadcastValidationFloatFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, float>;

TEST_SUITE(CL)
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
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Scale",{  2.f, 2.f, 2.f, -1.f, 1.f, 1.f})),
               framework::dataset::make("Expected", { true, true, false, false, false, false})),
               input1_info, input2_info, output_info, scale, expected)
{
    bool has_error = bool(CLPixelWiseMultiplication::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), scale, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(F16toF16)
TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF16Fixture<half_float::half>, PRECOMMIT, SmallShapes(), F16, F16, scale_255, TO_NEAREST_UP, EmptyActivationFunctionsDataset,
                                                 VALIDATE(float, 1.f))
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunWithActivation, ToF16Fixture<half_float::half>, ALL, TinyShapes(), F16, F16, scale_255, TO_NEAREST_UP, ActivationFunctionsDataset, VALIDATE(float,
                                                 1.f))
TEST_SUITE_END() // Scale255
TEST_SUITE_END() // F16toF16

TEST_SUITE(F32toF32)
TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, PRECOMMIT, SmallShapes(), F32, F32, scale_255, TO_NEAREST_UP, EmptyActivationFunctionsDataset, VALIDATE(float, 1.f))
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunWithActivation, ToF32Fixture<float>, ALL, TinyShapes(), F32, F32, scale_255, TO_NEAREST_UP, ActivationFunctionsDataset, VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255
TEST_SUITE_END() // F32toF32

PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, BroadcastFixture<float>, PRECOMMIT, SmallShapesBroadcast(), F32, F32, scale_255, TO_NEAREST_UP, EmptyActivationFunctionsDataset,
                                                 VALIDATE(float, 1.f))
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunWithActivationSmallBroadcast, BroadcastFixture<float>, ALL, TinyShapesBroadcast(), F32, F32, scale_255, TO_NEAREST_UP, ActivationFunctionsDataset,
                                                 VALIDATE(float, 1.f))

template <typename T>
using CLPixelWiseMultiplicationQuantizedFixture   = PixelWiseMultiplicationValidationQuantizedFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, T>;
using CLPixelWiseMultiplicationQSYMM16ToS32Fxture = PixelWiseMultiplicationValidationQuantizedFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, int16_t, int16_t, int32_t>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPixelWiseMultiplicationQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QASYMM8)),
                                                                                               framework::dataset::make("DataTypeIn2", DataType::QASYMM8)),
                                                                                       framework::dataset::make("DataTypeOut", DataType::QASYMM8)),
                                                                               framework::dataset::make("Scale", { 1.f, 2.f })),
                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                               framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_NEAREST_EVEN)),
                                                       framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                               framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                       framework::dataset::make("OUtQInfo", { QuantizationInfo(1.f / 255.f, 5) })),
                               InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPixelWiseMultiplicationQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QASYMM8_SIGNED)),
                                                                                               framework::dataset::make("DataTypeIn2", DataType::QASYMM8_SIGNED)),
                                                                                       framework::dataset::make("DataTypeOut", DataType::QASYMM8_SIGNED)),
                                                                               framework::dataset::make("Scale", { 1.f, 2.f })),
                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                               framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_NEAREST_EVEN)),
                                                       framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                               framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                       framework::dataset::make("OUtQInfo", { QuantizationInfo(1.f / 255.f, 5) })),
                               InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPixelWiseMultiplicationQuantizedFixture<int16_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QSYMM16)),
                                                                                               framework::dataset::make("DataTypeIn2", DataType::QSYMM16)),
                                                                                       framework::dataset::make("DataTypeOut", DataType::QSYMM16)),
                                                                               framework::dataset::make("Scale", { 1.f, 2.f })),
                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                               framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_NEAREST_EVEN)),
                                                       framework::dataset::make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0) })),
                                               framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0) })),
                                       framework::dataset::make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) })),
                               InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qsymm16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPixelWiseMultiplicationQuantizedFixture<int16_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QSYMM16)),
                                                                                               framework::dataset::make("DataTypeIn2", DataType::QSYMM16)),
                                                                                       framework::dataset::make("DataTypeOut", DataType::QSYMM16)),
                                                                               framework::dataset::make("Scale", { 1.f, 2.f })),
                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                               framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_NEAREST_EVEN)),
                                                       framework::dataset::make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0) })),
                                               framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0) })),
                                       framework::dataset::make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) })),
                               InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE(QSYMM16ToS32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPixelWiseMultiplicationQSYMM16ToS32Fxture, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                       framework::dataset::make("DataTypeIn1", DataType::QSYMM16)),
                                                                                               framework::dataset::make("DataTypeIn2", DataType::QSYMM16)),
                                                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                                                               framework::dataset::make("Scale", { 1.f })),
                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE })),
                                                               framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_NEAREST_EVEN)),
                                                       framework::dataset::make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0) })),
                                               framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0) })),
                                       framework::dataset::make("OutQInfo", { QuantizationInfo(1.f, 0) })),
                               InPlaceDataSet))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // QSYMM16ToS32
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // PixelWiseMultiplication
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
