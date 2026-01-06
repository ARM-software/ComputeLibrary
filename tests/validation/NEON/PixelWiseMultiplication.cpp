/*
 * Copyright (c) 2017-2021, 2024-2026 Arm Limited.
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

#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/DatatypeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/PixelWiseMultiplicationFixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"

#include <algorithm>
#include <tuple>

namespace arm_compute
{
namespace test
{
namespace validation
{

using framework::dataset::make;

namespace
{
const float scale_unity = 1.f;
const float scale_255   = 1.f / 255.f;
const float scale_other = 1.f / 32768.f;

constexpr AbsoluteTolerance<float> tolerance_qasymm8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for 8-bit quantized asymmetric data types */
constexpr AbsoluteTolerance<float> tolerance_qsymm16(
    1); /**< Tolerance value for comparing reference's output against implementation's output for 16-bit quantized symmetric data types */

const auto PixelWiseMultiplicationQSYMM16QuantDataset = combine(make("Src0QInfo", {QuantizationInfo(1.f / 32768.f, 0)}),
                                                                make("Src1QInfo", {QuantizationInfo(2.f / 32768.f, 0)}),
                                                                make("OutQInfo", {QuantizationInfo(5.f / 32768.f, 0)}));

const auto PixelWiseMultiplicationQASYMM8QuantDataset = combine(make("Src0QInfo", {QuantizationInfo(5.f / 32768.f, 0)}),
                                                                make("Src1QInfo", {QuantizationInfo(2.f / 32768.f, 0)}),
                                                                make("OutQInfo", {QuantizationInfo(1.f / 32768.f, 0)}));

const auto PixelWiseMultiplicationQASYMM8QuantInPlaceDataset =
    combine(make("Src0QInfo", {QuantizationInfo(5.f / 32768.f, 10)}),
            make("Src1QInfo", {QuantizationInfo(5.f / 32768.f, 10)}),
            make("OutQInfo", {QuantizationInfo(5.f / 32768.f, 10)}));

const auto PixelWiseMultiplicationPolicySTNUDataset =
    combine(make("ConvertPolicy", {ConvertPolicy::SATURATE}), make("RoundingPolicy", {RoundingPolicy::TO_NEAREST_UP}));

const auto PixelWiseMultiplicationPolicySTZDataset =
    combine(make("ConvertPolicy", {ConvertPolicy::SATURATE}), make("RoundingPolicy", {RoundingPolicy::TO_ZERO}));

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
const auto InPlaceDataSet    = make("InPlace", {false, true});
const auto OutOfPlaceDataSet = make("InPlace", {false});

#define DEFAULT_VALIDATE          validate(Accessor(_target), _reference);
#define VALIDATE(TYPE, TOLERANCE) validate(Accessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);
#define WRAP_VALIDATE(TYPE, TOLERANCE) \
    validate_wrap(Accessor(_target), _reference, AbsoluteTolerance<TYPE>(TOLERANCE), 0.f);

// clang-format off
#define PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, DT3, SCALE, RP, INPLACE_DATASET, VALIDATE) \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, NEPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                        \
                           combine(                                            \
                           datasets::SHAPES,                                                                              \
                           make("DataType1", DataType::DT1),                                         \
                           make("DataType2", DataType::DT2),                                         \
                           make("DataType3", DataType::DT3),                                         \
                           make("Scale", std::move(SCALE)),                                          \
                           datasets::ConvertPolicies(),                                                                  \
                           make("RoundingPolicy", RoundingPolicy::RP),                               \
                           (INPLACE_DATASET)))                                                                            \
    {                                                                                                                     \
        if((DataType::DT1 != DataType::F16 &&                                                                             \
            DataType::DT2 != DataType::F16 &&                                                                             \
            DataType::DT3 != DataType::F16) || CPUInfo::get().has_fp16())                                                 \
        {                                                                                                                 \
            VALIDATE                                                                                                      \
        }                                                                                                                 \
        else                                                                                                              \
        {                                                                                                                 \
            ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");                       \
            framework::ARM_COMPUTE_PRINT_WARNING();                                                                          \
        }                                                                                                                 \
    }

// clang-format on

void validate_data_types(DataType input1_dtype, DataType input2_dtype, DataType output_dtype)
{
    const auto input1 = TensorInfo(TensorShape(27U, 13U, 2U), 1, input1_dtype);
    const auto input2 = TensorInfo(TensorShape(27U, 13U, 2U), 1, input2_dtype);
    auto       output = TensorInfo(TensorShape(27U, 13U, 2U), 1, output_dtype);

    bool is_valid = static_cast<bool>(NEPixelWiseMultiplication::validate(
        &input1, &input2, &output, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));

    const auto supports = {
        std::make_tuple(DataType::F32, DataType::F32, DataType::F32),
        std::make_tuple(DataType::F16, DataType::F16, DataType::F16),
        std::make_tuple(DataType::U8, DataType::U8, DataType::U8),
        std::make_tuple(DataType::U8, DataType::U8, DataType::S16),
        std::make_tuple(DataType::U8, DataType::S16, DataType::S16),
        std::make_tuple(DataType::S16, DataType::U8, DataType::S16),
        std::make_tuple(DataType::S16, DataType::S16, DataType::S16),
        std::make_tuple(DataType::S32, DataType::S32, DataType::S32),
        std::make_tuple(DataType::QSYMM16, DataType::QSYMM16, DataType::QSYMM16),
        std::make_tuple(DataType::QSYMM16, DataType::QSYMM16, DataType::S32),
        std::make_tuple(DataType::QASYMM8, DataType::QASYMM8, DataType::QASYMM8),
        std::make_tuple(DataType::QASYMM8_SIGNED, DataType::QASYMM8_SIGNED, DataType::QASYMM8_SIGNED)};

    const auto                            config      = std::make_tuple(input1_dtype, input2_dtype, output_dtype);
    const std::initializer_list<DataType> dtypes_list = {input1_dtype, input2_dtype, output_dtype};

    bool expected = false;
    if (cpu_supports_dtypes(dtypes_list))
    {
        expected = (std::find(supports.begin(), supports.end(), config) != supports.end());
    }

    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

} // namespace

using NEPixelWiseMultiplicationQASYMM8Fixture =
    PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, uint8_t, uint8_t>;
using NEPixelWiseMultiplicationQASYMM8SignedFixture =
    PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, int8_t, int8_t>;
using NEPixelWiseMultiplicationQSYMM16Fixture =
    PixelWiseMultiplicationValidationQuantizedFixture<Tensor, Accessor, NEPixelWiseMultiplication, int16_t, int16_t>;
using NEPixelWiseMultiplicationQSYMM16ToS32Fixture =
    PixelWiseMultiplicationValidationQuantizedFixture<Tensor,
                                                      Accessor,
                                                      NEPixelWiseMultiplication,
                                                      int16_t,
                                                      int16_t,
                                                      int32_t>;
template <typename T>
using NEPixelWiseMultiplicationToU8Fixture =
    PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, uint8_t>;
template <typename T>
using NEPixelWiseMultiplicationToS16Fixture =
    PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, int16_t>;
template <typename T>
using NEPixelWiseMultiplicationToS32Fixture =
    PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, int32_t>;
template <typename T>
using NEPixelWiseMultiplicationToF16Fixture =
    PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, half_float::half>;
template <typename T>
using NEPixelWiseMultiplicationToF32Fixture =
    PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, float>;
using NEPixelWiseMultiplicationU8U8ToS16Fixture =
    PixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, uint8_t, uint8_t, int16_t>;
template <typename T>
using NEPixelWiseMultiplicationBroadcastFixture =
    PixelWiseMultiplicationBroadcastValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T, T>;
using NEPixelWiseMultiplicationBroadcastQASYMM8Fixture =
    PixelWiseMultiplicationBroadcastValidationQuantizedFixture<Tensor,
                                                               Accessor,
                                                               NEPixelWiseMultiplication,
                                                               uint8_t,
                                                               uint8_t>;
using NEPixelWiseMultiplicationBroadcastQASYMM8SignedFixture =
    PixelWiseMultiplicationBroadcastValidationQuantizedFixture<Tensor,
                                                               Accessor,
                                                               NEPixelWiseMultiplication,
                                                               int8_t,
                                                               int8_t>;
using NEPixelWiseMultiplicationBroadcastQSYMM16Fixture =
    PixelWiseMultiplicationBroadcastValidationQuantizedFixture<Tensor,
                                                               Accessor,
                                                               NEPixelWiseMultiplication,
                                                               int16_t,
                                                               int16_t>;
using NEPixelWiseMultiplicationBroadcastQSYMM16ToS32Fixture =
    PixelWiseMultiplicationBroadcastValidationQuantizedFixture<Tensor,
                                                               Accessor,
                                                               NEPixelWiseMultiplication,
                                                               int16_t,
                                                               int16_t,
                                                               int32_t>;
using NEPixelWiseMultiplicationBroadcastU8U8ToS16Fixture =
    PixelWiseMultiplicationBroadcastValidationFixture<Tensor,
                                                      Accessor,
                                                      NEPixelWiseMultiplication,
                                                      uint8_t,
                                                      uint8_t,
                                                      int16_t>;
using NEPixelWiseMultiplicationBroadcastToS16Fixture =
    PixelWiseMultiplicationBroadcastValidationFixture<Tensor,
                                                      Accessor,
                                                      NEPixelWiseMultiplication,
                                                      uint8_t,
                                                      int16_t,
                                                      int16_t>;

TEST_SUITE(NEON)
TEST_SUITE(PixelWiseMultiplication)

// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),                 //1 Ok
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
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),                //14 S32 does not support scale255
                                                      }),
               make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
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
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                     }),
               make("Scale",{  scale_unity,
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
                                                   scale_unity,
                                                   scale_255}),
               make("OverflowPolicy",{
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
                                                   ConvertPolicy::SATURATE,
                                        }),
               make("Expected", { true, true, true, false, false, false, false, false, true , false, false, true, false, false})
               ),
               input1_info, input2_info, output_info, scale, policy, expected)
{
    bool has_error = bool(NEPixelWiseMultiplication::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), scale, policy, RoundingPolicy::TO_ZERO));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on

/// @note: Do not modify. Validating all data types is pretty fast.
DATA_TEST_CASE(ValidateAllDataTypes,
               framework::DatasetMode::ALL,
               combine(datasets::AllDataTypes("Input1DataType"),
                       datasets::AllDataTypes("Input2DataType"),
                       datasets::AllDataTypes("OutputDataType")),
               input1_dtype,
               input2_dtype,
               output_dtype)
{
    validate_data_types(input1_dtype, input2_dtype, output_dtype);
}

TEST_SUITE(InPlaceValidate)
TEST_CASE(SingleTensor, framework::DatasetMode::ALL)
{
    const auto random_shape       = TensorShape{9, 9};
    const auto single_tensor_info = TensorInfo{random_shape, 1, DataType::F32};

    Status result = NEPixelWiseMultiplication::validate(&single_tensor_info, &single_tensor_info, &single_tensor_info,
                                                        scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(ValidBroadCast, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{27U, 13U, 2U};
    const auto smaller_shape = TensorShape{1U, 13U, 2U};

    const auto larger_tensor_info  = TensorInfo{larger_shape, 1, DataType::F32};
    const auto smaller_tensor_info = TensorInfo{smaller_shape, 1, DataType::F32};

    Status result = NEPixelWiseMultiplication::validate(&larger_tensor_info, &smaller_tensor_info, &larger_tensor_info,
                                                        scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastOutput, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{27U, 13U, 2U};
    const auto smaller_shape = TensorShape{1U, 13U, 2U};

    const auto larger_tensor_info  = TensorInfo{larger_shape, 1, DataType::F32};
    const auto smaller_tensor_info = TensorInfo{smaller_shape, 1, DataType::F32};

    Status result = NEPixelWiseMultiplication::validate(&larger_tensor_info, &smaller_tensor_info, &smaller_tensor_info,
                                                        scale_unity, ConvertPolicy::WRAP, RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastBoth, framework::DatasetMode::ALL)
{
    const auto shape0 = TensorShape{9U, 9U};
    const auto shape1 = TensorShape{9U, 1U, 2U};

    const auto info0 = TensorInfo{shape0, 1, DataType::F32};
    const auto info1 = TensorInfo{shape1, 1, DataType::F32};

    Status result{};

    result = NEPixelWiseMultiplication::validate(&info0, &info1, &info0, scale_unity, ConvertPolicy::WRAP,
                                                 RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEPixelWiseMultiplication::validate(&info0, &info1, &info1, scale_unity, ConvertPolicy::WRAP,
                                                 RoundingPolicy::TO_ZERO);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // InPlaceValidate

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
TEST_SUITE(ScaleUnity)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQASYMM8SignedFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QASYMM8_SIGNED),
                               make("DataTypeIn2", DataType::QASYMM8_SIGNED),
                               make("DataTypeOut", DataType::QASYMM8_SIGNED),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
#ifdef ARM_COMPUTE_ENABLE_SME2
FIXTURE_DATA_TEST_CASE(RunSMEMul,
                       NEPixelWiseMultiplicationQASYMM8SignedFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SMEMulShapes(),
                               make("DataTypeIn1", DataType::QASYMM8_SIGNED),
                               make("DataTypeIn2", DataType::QASYMM8_SIGNED),
                               make("DataTypeOut", DataType::QASYMM8_SIGNED),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
#endif // ARM_COMPUTE_ENABLE_SME2
FIXTURE_DATA_TEST_CASE(RunSmallInPlace,
                       NEPixelWiseMultiplicationQASYMM8SignedFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QASYMM8_SIGNED),
                               make("DataTypeIn2", DataType::QASYMM8_SIGNED),
                               make("DataTypeOut", DataType::QASYMM8_SIGNED),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantInPlaceDataset,
                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Broadcast)
#ifdef ARM_COMPUTE_ENABLE_SME2
FIXTURE_DATA_TEST_CASE(RunSMEMul,
                       NEPixelWiseMultiplicationBroadcastQASYMM8SignedFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SMEMulShapesBroadcast(),
                               make("DataTypeIn1", DataType::QASYMM8_SIGNED),
                               make("DataTypeIn2", DataType::QASYMM8_SIGNED),
                               make("DataTypeOut", DataType::QASYMM8_SIGNED),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
#endif           // ARM_COMPUTE_ENABLE_SME2
TEST_SUITE_END() // Broadcast
TEST_SUITE_END() // ScaleUnity
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QASYMM8)
TEST_SUITE(Scale255)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQASYMM8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QASYMM8),
                               make("DataTypeIn2", DataType::QASYMM8),
                               make("DataTypeOut", DataType::QASYMM8),
                               make("Scale", {scale_255}),
                               PixelWiseMultiplicationPolicySTNUDataset,
                               PixelWiseMultiplicationQASYMM8QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Scale255
TEST_SUITE(ScaleUnity)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQASYMM8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QASYMM8),
                               make("DataTypeIn2", DataType::QASYMM8),
                               make("DataTypeOut", DataType::QASYMM8),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // ScaleUnity
TEST_SUITE(ScaleOther)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQASYMM8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QASYMM8),
                               make("DataTypeIn2", DataType::QASYMM8),
                               make("DataTypeOut", DataType::QASYMM8),
                               make("Scale", {scale_other}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // ScaleOther
TEST_SUITE(Broadcast)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationBroadcastQASYMM8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataTypeIn1", DataType::QASYMM8),
                               make("DataTypeIn2", DataType::QASYMM8),
                               make("DataTypeOut", DataType::QASYMM8),
                               make("Scale", {scale_other}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunTinyInPlace,
                       NEPixelWiseMultiplicationBroadcastQASYMM8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::TinyShapesBroadcastInplace(),
                               make("DataTypeIn1", DataType::QASYMM8),
                               make("DataTypeIn2", DataType::QASYMM8),
                               make("DataTypeOut", DataType::QASYMM8),
                               make("Scale", {scale_other}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQASYMM8QuantInPlaceDataset,
                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Broadcast
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QSYMM16)
TEST_SUITE(Scale255)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQSYMM16Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QSYMM16),
                               make("DataTypeIn2", DataType::QSYMM16),
                               make("DataTypeOut", DataType::QSYMM16),
                               make("Scale", {scale_255}),
                               PixelWiseMultiplicationPolicySTNUDataset,
                               PixelWiseMultiplicationQSYMM16QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // Scale255
TEST_SUITE(ScaleUnity)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQSYMM16Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QSYMM16),
                               make("DataTypeIn2", DataType::QSYMM16),
                               make("DataTypeOut", DataType::QSYMM16),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQSYMM16QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // ScaleUnity
TEST_SUITE(ScaleOther)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQSYMM16Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QSYMM16),
                               make("DataTypeIn2", DataType::QSYMM16),
                               make("DataTypeOut", DataType::QSYMM16),
                               make("Scale", {scale_other}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQSYMM16QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}

TEST_SUITE_END() // ScaleOther
TEST_SUITE(NonXBroadcast)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationBroadcastQSYMM16Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesNonXBroadcast(),
                               make("DataTypeIn1", DataType::QSYMM16),
                               make("DataTypeIn2", DataType::QSYMM16),
                               make("DataTypeOut", DataType::QSYMM16),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQSYMM16QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // NonXBroadcast
TEST_SUITE_END() // QSYMM16
TEST_SUITE(QSYMM16toS32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationQSYMM16ToS32Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::QSYMM16),
                               make("DataTypeIn2", DataType::QSYMM16),
                               make("DataTypeOut", DataType::S32),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQSYMM16QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE(NonXBroadcast)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationBroadcastQSYMM16ToS32Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesNonXBroadcast(),
                               make("DataTypeIn1", DataType::QSYMM16),
                               make("DataTypeIn2", DataType::QSYMM16),
                               make("DataTypeOut", DataType::S32),
                               make("Scale", {scale_unity}),
                               PixelWiseMultiplicationPolicySTZDataset,
                               PixelWiseMultiplicationQSYMM16QuantDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // NonXBroadcast
TEST_SUITE_END() // QSYMM16toS32
TEST_SUITE_END() // Quantized

TEST_SUITE(U8U8toS16)

FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationU8U8ToS16Fixture,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::U8),
                               make("DataTypeIn2", DataType::U8),
                               make("DataTypeOut", DataType::S16),
                               make("Scale", {scale_255}),
                               datasets::ConvertPolicies(),
                               make("RoundingPolicy", RoundingPolicy::TO_NEAREST_UP),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate_wrap(Accessor(_target), _reference, AbsoluteTolerance<int16_t>(1), 0.f);
}

TEST_SUITE(NonXBroadcast)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPixelWiseMultiplicationBroadcastU8U8ToS16Fixture,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapesNonXBroadcast(),
                               make("DataTypeIn1", DataType::U8),
                               make("DataTypeIn2", DataType::U8),
                               make("DataTypeOut", DataType::S16),
                               make("Scale", {scale_255}),
                               datasets::ConvertPolicies(),
                               make("RoundingPolicy", RoundingPolicy::TO_NEAREST_UP),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate_wrap(Accessor(_target), _reference, AbsoluteTolerance<int16_t>(1), 0.f);
}
TEST_SUITE_END() // NonXBroadcast

FIXTURE_DATA_TEST_CASE(RunSmall1,
                       NEPixelWiseMultiplicationU8U8ToS16Fixture,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataTypeIn1", DataType::U8),
                               make("DataTypeIn2", DataType::U8),
                               make("DataTypeOut", DataType::S16),
                               make("Scale", {scale_other}),
                               datasets::ConvertPolicies(),
                               make("RoundingPolicy", RoundingPolicy::TO_ZERO),
                               make("InPlace", {false})))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END() // U8U8toS16

TEST_SUITE(U8toU8)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToU8Fixture<uint8_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 U8,
                                                 U8,
                                                 U8,
                                                 scale_255,
                                                 TO_NEAREST_UP,
                                                 InPlaceDataSet,
                                                 WRAP_VALIDATE(uint8_t, 1))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToU8Fixture<uint8_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 U8,
                                                 U8,
                                                 U8,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE(NonXBroadcast)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 BroadcastFixture<uint8_t>,
                                                 ALL,
                                                 SmallShapesNonXBroadcast(),
                                                 U8,
                                                 U8,
                                                 U8,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 OutOfPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // NonXBroadcast
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToU8Fixture<uint8_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 U8,
                                                 U8,
                                                 U8,
                                                 scale_other,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // U8toU8

TEST_SUITE(U8toS16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS16Fixture<uint8_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 U8,
                                                 S16,
                                                 S16,
                                                 scale_255,
                                                 TO_NEAREST_UP,
                                                 OutOfPlaceDataSet,
                                                 WRAP_VALIDATE(int16_t, 2))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS16Fixture<uint8_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 U8,
                                                 S16,
                                                 S16,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 OutOfPlaceDataSet,
                                                 DEFAULT_VALIDATE)

TEST_SUITE(NonXBroadcast)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 BroadcastToS16Fixture,
                                                 ALL,
                                                 SmallShapesNonXBroadcast(),
                                                 U8,
                                                 S16,
                                                 S16,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 OutOfPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // NonXBroadcast
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS16Fixture<uint8_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 U8,
                                                 S16,
                                                 S16,
                                                 scale_other,
                                                 TO_ZERO,
                                                 OutOfPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // U8toS16

TEST_SUITE(S16toS16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS16Fixture<int16_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 S16,
                                                 S16,
                                                 S16,
                                                 scale_255,
                                                 TO_NEAREST_UP,
                                                 InPlaceDataSet,
                                                 WRAP_VALIDATE(int16_t, 2))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS16Fixture<int16_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 S16,
                                                 S16,
                                                 S16,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE(NonXBroadcast)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 BroadcastFixture<int16_t>,
                                                 ALL,
                                                 SmallShapesNonXBroadcast(),
                                                 S16,
                                                 S16,
                                                 S16,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 OutOfPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // NonXBroadcast

TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS16Fixture<int16_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 S16,
                                                 S16,
                                                 S16,
                                                 scale_other,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // S16toS16

TEST_SUITE(S32toS32)

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS32Fixture<int32_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 S32,
                                                 S32,
                                                 S32,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 WRAP_VALIDATE(int32_t, 1))
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToS32Fixture<int32_t>,
                                                 ALL,
                                                 SmallShapes(),
                                                 S32,
                                                 S32,
                                                 S32,
                                                 scale_other,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 WRAP_VALIDATE(int32_t, 1))
TEST_SUITE_END() // ScaleOther

TEST_SUITE(Broadcast)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 BroadcastFixture<int32_t>,
                                                 ALL,
                                                 SmallShapesBroadcast(),
                                                 S32,
                                                 S32,
                                                 S32,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 make("InPlace", {false}),
                                                 WRAP_VALIDATE(int32_t, 1))
TEST_SUITE_END() // Broadcast

TEST_SUITE_END() // S32toS32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16toF16)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToF16Fixture<half_float::half>,
                                                 ALL,
                                                 SmallShapes(),
                                                 F16,
                                                 F16,
                                                 F16,
                                                 scale_255,
                                                 TO_NEAREST_UP,
                                                 InPlaceDataSet,
                                                 VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE_END() // F16toF16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(F32toF32)

TEST_SUITE(Scale255)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToF32Fixture<float>,
                                                 ALL,
                                                 SmallShapes(),
                                                 F32,
                                                 F32,
                                                 F32,
                                                 scale_255,
                                                 TO_NEAREST_UP,
                                                 InPlaceDataSet,
                                                 VALIDATE(float, 1.f))
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToF32Fixture<float>,
                                                 ALL,
                                                 SmallShapes(),
                                                 F32,
                                                 F32,
                                                 F32,
                                                 scale_unity,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 ToF32Fixture<float>,
                                                 ALL,
                                                 SmallShapes(),
                                                 F32,
                                                 F32,
                                                 F32,
                                                 scale_other,
                                                 TO_ZERO,
                                                 InPlaceDataSet,
                                                 DEFAULT_VALIDATE)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // F32toF32

TEST_SUITE(Broadcast)
PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall,
                                                 BroadcastFixture<float>,
                                                 ALL,
                                                 SmallShapesBroadcast(),
                                                 F32,
                                                 F32,
                                                 F32,
                                                 scale_255,
                                                 TO_NEAREST_UP,
                                                 make("InPlace", {false}),
                                                 VALIDATE(float, 1.f))
TEST_SUITE_END() // Broadcast

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
