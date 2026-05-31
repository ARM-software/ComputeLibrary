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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/DatatypeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/ArithmeticOperationsFixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{

using framework::dataset::make;

namespace
{
#ifdef __aarch64__
constexpr AbsoluteTolerance<float> tolerance_qasymm8(
    0); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
#else   //__aarch64__
constexpr AbsoluteTolerance<float> tolerance_qasymm8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
#endif  //__aarch64__
constexpr AbsoluteTolerance<int16_t> tolerance_qsymm16(
    1); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */

// Quantization Infomation DataSet
const auto ArithmeticSubtractionQuantizationInfoDataset =
    combine(make("QuantizationInfoIn1", {QuantizationInfo(10, 120)}),
            make("QuantizationInfoIn2", {QuantizationInfo(20, 110)}),
            make("QuantizationInfoOut", {QuantizationInfo(15, 125)}));
const auto ArithmeticSubtractionQuantizationInfoSignedDataset =
    combine(make("QuantizationInfoIn1", {QuantizationInfo(0.5f, 10)}),
            make("QuantizationInfoIn2", {QuantizationInfo(0.5f, 20)}),
            make("QuantizationInfoOut", {QuantizationInfo(0.5f, 50)}));
const auto ArithmeticSubtractionQuantizationInfoSignedInPlaceDataset =
    combine(make("QuantizationInfoIn1", {QuantizationInfo(0.8f, 10)}),
            make("QuantizationInfoIn2", {QuantizationInfo(0.8f, 10)}),
            make("QuantizationInfoOut", {QuantizationInfo(0.8f, 10)}));
const auto ArithmeticSubtractionQuantizationInfo16bitSymmetric = combine(
    make("QuantizationInfoIn1", {QuantizationInfo(0.003f, 0)}),
    make("QuantizationInfoIn2", {QuantizationInfo(0.007f, 0)}),
    make("QuantizationInfoOut", {QuantizationInfo(0.2f, 0), QuantizationInfo(0.002f, 0) /* for more saturation */}));

const auto InPlaceDataSet    = make("InPlace", {false, true});
const auto OutOfPlaceDataSet = make("InPlace", {false});

void validate_data_types(DataType input1_dtype, DataType input2_dtype, DataType output_dtype)
{
    const auto input1 = TensorInfo(TensorShape(27U, 13U, 2U), 1, input1_dtype);
    const auto input2 = TensorInfo(TensorShape(27U, 13U, 2U), 1, input2_dtype);
    auto       output = TensorInfo(TensorShape(27U, 13U, 2U), 1, output_dtype);

    const bool is_valid =
        static_cast<bool>(NEArithmeticSubtraction::validate(&input1, &input2, &output, ConvertPolicy::SATURATE));

    const auto supports = {
        std::make_tuple(DataType::F32, DataType::F32, DataType::F32),
        std::make_tuple(DataType::F16, DataType::F16, DataType::F16),
        std::make_tuple(DataType::U8, DataType::U8, DataType::U8),
        std::make_tuple(DataType::S16, DataType::S16, DataType::S16),
        std::make_tuple(DataType::S32, DataType::S32, DataType::S32),
        std::make_tuple(DataType::QSYMM16, DataType::QSYMM16, DataType::QSYMM16),
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

TEST_SUITE(NEON)
TEST_SUITE(ArithmeticSubtraction)

template <typename T>
using NEArithmeticSubtractionFixture =
    ArithmeticSubtractionValidationFixture<Tensor, Accessor, NEArithmeticSubtraction, T>;
template <typename T>
using NEArithmeticSubtractionBroadcastFixture =
    ArithmeticSubtractionBroadcastValidationFixture<Tensor, Accessor, NEArithmeticSubtraction, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
        make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                 TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::QASYMM8), // Mismatching types
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8), // Invalid convert policy
        }),
        make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
        }),
        make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
        }),
        make("ConvertPolicy",{ ConvertPolicy::SATURATE,
                                                   ConvertPolicy::SATURATE,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
                                                   ConvertPolicy::WRAP,
        }),
        make("Expected", { true, false, false, false, false})
        ),
        input1_info, input2_info, output_info, policy, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEArithmeticSubtraction::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), policy)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(InPlaceValidate)
TEST_CASE(SingleTensor, framework::DatasetMode::ALL)
{
    const auto random_shape       = TensorShape{9, 9};
    const auto single_tensor_info = TensorInfo{random_shape, 1, DataType::F32};

    Status result = NEArithmeticSubtraction::validate(&single_tensor_info, &single_tensor_info, &single_tensor_info,
                                                      ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(ValidBroadCast, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{27U, 13U, 2U};
    const auto smaller_shape = TensorShape{1U, 13U, 2U};

    const auto larger_tensor_info  = TensorInfo{larger_shape, 1, DataType::F32};
    const auto smaller_tensor_info = TensorInfo{smaller_shape, 1, DataType::F32};

    Status result = NEArithmeticSubtraction::validate(&larger_tensor_info, &smaller_tensor_info, &larger_tensor_info,
                                                      ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastOutput, framework::DatasetMode::ALL)
{
    const auto larger_shape  = TensorShape{27U, 13U, 2U};
    const auto smaller_shape = TensorShape{1U, 13U, 2U};

    const auto larger_tensor_info  = TensorInfo{larger_shape, 1, DataType::F32};
    const auto smaller_tensor_info = TensorInfo{smaller_shape, 1, DataType::F32};

    Status result = NEArithmeticSubtraction::validate(&larger_tensor_info, &smaller_tensor_info, &smaller_tensor_info,
                                                      ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidBroadcastBoth, framework::DatasetMode::ALL)
{
    const auto shape0 = TensorShape{9U, 9U};
    const auto shape1 = TensorShape{9U, 1U, 2U};

    const auto info0 = TensorInfo{shape0, 1, DataType::F32};
    const auto info1 = TensorInfo{shape1, 1, DataType::F32};

    Status result{};

    result = NEArithmeticSubtraction::validate(&info0, &info1, &info0, ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEArithmeticSubtraction::validate(&info0, &info1, &info1, ConvertPolicy::WRAP);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // InPlaceValidate

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

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::U8),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEArithmeticSubtractionBroadcastFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataType", DataType::U8),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

using NEArithmeticSubtractionQASYMM8Fixture =
    ArithmeticSubtractionValidationQuantizedFixture<Tensor, Accessor, NEArithmeticSubtraction, uint8_t>;
using NEArithmeticSubtractionQASYMM8SignedFixture =
    ArithmeticSubtractionValidationQuantizedFixture<Tensor, Accessor, NEArithmeticSubtraction, int8_t>;
using NEArithmeticSubtractionQSYMM16Fixture =
    ArithmeticSubtractionValidationQuantizedFixture<Tensor, Accessor, NEArithmeticSubtraction, int16_t>;

template <typename T>
using NEArithmeticSubtractionQuantizedBroadcastFixture =
    ArithmeticSubtractionValidationQuantizedBroadcastFixture<Tensor, Accessor, NEArithmeticSubtraction, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionQASYMM8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE}),
                               ArithmeticSubtractionQuantizationInfoDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEArithmeticSubtractionQuantizedBroadcastFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataType", DataType::QASYMM8),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE}),
                               ArithmeticSubtractionQuantizationInfoDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionQASYMM8SignedFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE}),
                               ArithmeticSubtractionQuantizationInfoSignedDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEArithmeticSubtractionQuantizedBroadcastFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE}),
                               ArithmeticSubtractionQuantizationInfoSignedDataset,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunTinyBroadcastInPlace,
                       NEArithmeticSubtractionQuantizedBroadcastFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::TinyShapesBroadcastInplace(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE}),
                               ArithmeticSubtractionQuantizationInfoSignedInPlaceDataset,
                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionQSYMM16Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QSYMM16),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE}),
                               ArithmeticSubtractionQuantizationInfo16bitSymmetric,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEArithmeticSubtractionQuantizedBroadcastFixture<int16_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataType", DataType::QSYMM16),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE}),
                               ArithmeticSubtractionQuantizationInfo16bitSymmetric,
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionFixture<int16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::S16),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEArithmeticSubtractionBroadcastFixture<int16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataType", DataType::S16),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArithmeticSubtractionFixture<int16_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(),
                               make("DataType", DataType::S16),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionFixture<int32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::S32),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEArithmeticSubtractionBroadcastFixture<int32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataType", DataType::S32),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArithmeticSubtractionFixture<int32_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(),
                               make("DataType", DataType::S32),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::F16),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // F16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArithmeticSubtractionFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::F32),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArithmeticSubtractionFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(),
                               make("DataType", DataType::F32),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast,
                       NEArithmeticSubtractionBroadcastFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapesBroadcast(),
                               make("DataType", DataType::F32),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast,
                       NEArithmeticSubtractionBroadcastFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapesBroadcast(),
                               make("DataType", DataType::F32),
                               make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
                               OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ArithmeticSubtraction
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
