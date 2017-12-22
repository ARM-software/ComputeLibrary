/*
 * Copyright (c) 2017 ARM Limited.
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
#include "tests/validation/fixtures/FixedPointPixelWiseMultiplicationFixture.h"
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

// *INDENT-OFF*
// clang-format off
#define VALIDATE(TYPE, TOLERANCE) validate(CLAccessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);

#define PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, SCALE, RP, VALIDATE) \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, CLPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                   \
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

#define FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, SCALE, RP, FPP_START, FPP_END) \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, CLFixedPointPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                      \
                           combine(combine(combine(combine(combine(combine(                                                            \
                           datasets::SHAPES,                                                                                           \
                           framework::dataset::make("DataType1", DataType::DT1)),                                                      \
                           framework::dataset::make("DataType2", DataType::DT2)),                                                      \
                           framework::dataset::make("Scale", std::move(SCALE))),                                                       \
                           datasets::ConvertPolicies()),                                                                               \
                           framework::dataset::make("RoundingPolicy", RoundingPolicy::RP)),                                            \
                           framework::dataset::make("FixedPointPosition", FPP_START, FPP_END)))                                        \
    {                                                                                                                                  \
        validate(CLAccessor(_target), _reference);                                                                                     \
    }
// clang-format on
// *INDENT-ON*
} // namespace

template <typename T>
using CLPixelWiseMultiplicationToF16Fixture = PixelWiseMultiplicationValidationFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, half_float::half>;
template <typename T>
using CLPixelWiseMultiplicationToF32Fixture = PixelWiseMultiplicationValidationFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, float>;
template <typename T>
using CLPixelWiseMultiplicationToQS8Fixture = PixelWiseMultiplicationValidationFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, qint8_t>;
template <typename T>
using CLPixelWiseMultiplicationToQS16Fixture = PixelWiseMultiplicationValidationFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T, qint16_t>;
template <typename T>
using CLFixedPointPixelWiseMultiplicationFixture = FixedPointPixelWiseMultiplicationValidationFixture<CLTensor, CLAccessor, CLPixelWiseMultiplication, T>;

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
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),  // Mismatching data type
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),  // Mismatching fixed point
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),  // Invalid scale
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS16, 2),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),
                                                     })),
               framework::dataset::make("Scale",{  2.f, 2.f, 2.f, -1.f, 1.f, 1.f, 1.f, 1.f, 3.f})),
               framework::dataset::make("Expected", { true, true, false, false, false, false, false, false, false })),
               input1_info, input2_info, output_info, scale, expected)
{
    bool has_error = bool(CLPixelWiseMultiplication::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), scale, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(F16toF16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF16Fixture<half_float::half>, PRECOMMIT, SmallShapes(), F16, F16, scale_255, TO_NEAREST_UP, VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE_END() // F16toF16

TEST_SUITE(F32toF32)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, ToF32Fixture<float>, PRECOMMIT, SmallShapes(), F32, F32, scale_255, TO_NEAREST_UP, VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE_END() // F32toF32

TEST_SUITE_END() // PixelWiseMultiplication

TEST_SUITE(FixedPointPixelWiseMultiplication)

TEST_SUITE(QS8)

TEST_SUITE(ScaleUnity)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, scale_unity, TO_ZERO, 1, 7)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE_END() // QS8

TEST_SUITE(QS16)

TEST_SUITE(ScaleUnity)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, scale_unity, TO_ZERO, 1, 15)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE_END() // QS16

TEST_SUITE_END() // FixedPointPixelWiseMultiplication
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
