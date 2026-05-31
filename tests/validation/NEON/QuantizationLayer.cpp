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
#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/QuantizationLayerFixture.h"
#include "tests/validation/Validation.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
/** Tolerance for quantization */
/// @note: We do not expect any difference between our reference and target implementations for UInt8 and Int8
constexpr AbsoluteTolerance<uint8_t> tolerance_u8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for QASYMM8 data types */
constexpr AbsoluteTolerance<int8_t> tolerance_s8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for QASYMM8_SIGNED data types */
constexpr AbsoluteTolerance<int8_t> zero_tolerance_s8(0);

constexpr AbsoluteTolerance<uint16_t> tolerance_u16(
    1); /**< Tolerance value for comparing reference's output against implementation's output for QASYMM16 data types */
const auto QuantizationSmallShapes = concat(datasets::Small3DShapes(), datasets::Small4DShapes());
const auto QuantizationLargeShapes = concat(datasets::Large3DShapes(), datasets::Large4DShapes());
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(QuantizationLayer)

TEST_CASE(ProperlyRoundedRequantization, framework::DatasetMode::ALL)
{
    // The test case here covers both Int8 and UInt8 because the underlying kernel is the same
    const auto shape     = TensorShape(18U); // > 16 for channel dim. to stress vector and leftover loops
    const auto dtype     = DataType::QASYMM8_SIGNED;
    const auto in_qinfo  = QuantizationInfo(0.5f, -1);
    const auto out_qinfo = QuantizationInfo(1.f, -1);

    Tensor input  = create_tensor<Tensor>(shape, dtype, 1, in_qinfo);
    Tensor output = create_tensor<Tensor>(shape, dtype, 1, out_qinfo);

    NEQuantizationLayer quant_layer;
    quant_layer.configure(&input, &output);

    input.allocator()->allocate();
    output.allocator()->allocate();

    std::vector<int8_t> values   = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35};
    std::vector<int8_t> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}; // (x + 1)/2 - 1

    SimpleTensor<int8_t> ref{shape, dtype, 1, out_qinfo};

    ARM_COMPUTE_EXPECT(values.size() == shape.x(), framework::LogLevel::ERRORS);

    library->fill_static_values(Accessor(input), values);
    library->fill_static_values(ref, expected);

    quant_layer.run();

    validate(Accessor(output), ref, zero_tolerance_s8);
}

TEST_CASE(QSymm8_per_channel_validate_scales, framework::DatasetMode::ALL)
{
    // In this test we make sure validate does not raise an error when we pass a properly initialized vector of scales matching
    // the number of channels
    const auto         input_info  = TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32);
    auto               output_info = TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM8_PER_CHANNEL);
    Tensor             input       = create_tensor<Tensor>(input_info);
    std::vector<float> scale(16, 0.5f);
    Tensor             output =
        create_tensor<Tensor>(output_info.tensor_shape(), DataType::QSYMM8_PER_CHANNEL, 1, QuantizationInfo(scale));
    ARM_COMPUTE_EXPECT(bool(NEQuantizationLayer::validate(&input.info()->clone()->set_is_resizable(false),
                                                          &output.info()->clone()->set_is_resizable(false))) == true,
                       framework::LogLevel::ERRORS);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("InputInfo", { TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Wrong output data type
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),  // Wrong output data type
                                                       TensorInfo(TensorShape(16U, 16U, 2U, 5U), 1, DataType::F32),  // Missmatching shapes
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),  // Valid
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // PER_CHANNEL only supported for F32
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM8),  // PER_CHANNEL only supported for F32
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM16),  // PER_CHANNEL only supported for F32
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F16),  // PER_CHANNEL only supported for F32
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32), // Quantization info's scales not initialized
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::U16),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM8_PER_CHANNEL),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM8_PER_CHANNEL),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM8_PER_CHANNEL),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM8_PER_CHANNEL),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QSYMM8_PER_CHANNEL),
                                                     }),
               make("Expected", { false, false, false, true,false,false,false,false,false})
               ),
               input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEQuantizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEQuantizationLayerQASYMM8Fixture =
    QuantizationValidationFixture<Tensor, Accessor, NEQuantizationLayer, T, uint8_t>;
template <typename T>
using NEQuantizationLayerQASYMM8SignedFixture =
    QuantizationValidationFixture<Tensor, Accessor, NEQuantizationLayer, T, int8_t>;
template <typename T>
using NEQuantizationLayerQASYMM16Fixture =
    QuantizationValidationFixture<Tensor, Accessor, NEQuantizationLayer, T, uint16_t>;
template <typename T>
using NEQuantizationLayerQSYMM8_PER_CHANNEL_Fixture =
    QuantizationValidationFixture<Tensor, Accessor, NEQuantizationLayer, T, int8_t>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8,
                       NEQuantizationLayerQASYMM8Fixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::F32),
                               make("DataTypeOut", {DataType::QASYMM8}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8Signed,
                       NEQuantizationLayerQASYMM8SignedFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::F32),
                               make("DataTypeOut", {DataType::QASYMM8_SIGNED}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM16,
                       NEQuantizationLayerQASYMM16Fixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::F32),
                               make("DataTypeOut", {DataType::QASYMM16}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u16);
}
FIXTURE_DATA_TEST_CASE(RunLargeQASYMM8,
                       NEQuantizationLayerQASYMM8Fixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(QuantizationLargeShapes,
                               make("DataType", DataType::F32),
                               make("DataTypeOut", {DataType::QASYMM8}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunLargeQASYMM16,
                       NEQuantizationLayerQASYMM16Fixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(QuantizationLargeShapes,
                               make("DataType", DataType::F32),
                               make("DataTypeOut", {DataType::QASYMM16}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u16);
}

FIXTURE_DATA_TEST_CASE(RunSmallQSYMM8_PER_CHANNEL,
                       NEQuantizationLayerQSYMM8_PER_CHANNEL_Fixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::F32),
                               make("DataTypeOut", {DataType::QSYMM8_PER_CHANNEL}),
                               make("QuantizationInfoIgnored", {QuantizationInfo()})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}

TEST_SUITE_END() // FP32
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8,
                       NEQuantizationLayerQASYMM8Fixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::F16),
                               make("DataTypeOut", {DataType::QASYMM8}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_u8);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8Signed,
                       NEQuantizationLayerQASYMM8SignedFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::F16),
                               make("DataTypeOut", {DataType::QASYMM8_SIGNED}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_s8);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM16,
                       NEQuantizationLayerQASYMM16Fixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::F16),
                               make("DataTypeOut", {DataType::QASYMM16}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_u16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLargeQASYMM8,
                       NEQuantizationLayerQASYMM8Fixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(QuantizationLargeShapes,
                               make("DataType", DataType::F16),
                               make("DataTypeOut", {DataType::QASYMM8}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_u8);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLargeQASYMM16,
                       NEQuantizationLayerQASYMM16Fixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(QuantizationLargeShapes,
                               make("DataType", DataType::F16),
                               make("DataTypeOut", {DataType::QASYMM16}),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_u16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           //  ARM_COMPUTE_ENABLE_FP16
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
template <typename T>
using NEQuantizationLayerQASYMM8GenFixture =
    QuantizationValidationGenericFixture<Tensor, Accessor, NEQuantizationLayer, T, uint8_t>;
template <typename T>
using NEQuantizationLayerQASYMM8_SIGNEDGenFixture =
    QuantizationValidationGenericFixture<Tensor, Accessor, NEQuantizationLayer, T, int8_t>;
template <typename T>
using NEQuantizationLayerQASYMM16GenFixture =
    QuantizationValidationGenericFixture<Tensor, Accessor, NEQuantizationLayer, T, uint16_t>;
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8,
                       NEQuantizationLayerQASYMM8GenFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::QASYMM8),
                               make("DataTypeOut", {DataType::QASYMM8}),
                               make("QuantizationInfoOutput", {QuantizationInfo(0.5f, 10)}),
                               make("QuantizationInfoInput", {QuantizationInfo(2.0f, 15), QuantizationInfo(0.5f, 25)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(ConvertUint8toInt8,
                       NEQuantizationLayerQASYMM8GenFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(QuantizationSmallShapes,
                               make("DataType", DataType::QASYMM8),
                               make("DataTypeOut", {DataType::QASYMM8_SIGNED}),
                               make("QuantizationInfoOutput", {QuantizationInfo(2.0f, -1)}),
                               make("QuantizationInfoInput", {QuantizationInfo(2.0f, 127)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(
    RunSmallQASYMM8_SIGNED,
    NEQuantizationLayerQASYMM8_SIGNEDGenFixture<uint8_t>,
    framework::DatasetMode::ALL,
    combine(QuantizationSmallShapes,
            make("DataTypeIn", DataType::QASYMM8),
            make("DataTypeOut", {DataType::QASYMM8_SIGNED}),
            make("QuantizationInfoOutput", {QuantizationInfo(1.0f, 10), QuantizationInfo(2.0f, -25)}),
            make("QuantizationInfoInput", {QuantizationInfo(1.0f, 15), QuantizationInfo(1.0f, 127)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM16,
                       NEQuantizationLayerQASYMM16GenFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(QuantizationSmallShapes,
                               make("DataTypeIn", DataType::QASYMM8),
                               make("DataTypeOut", {DataType::QASYMM16}),
                               make("QuantizationInfoOutput", {QuantizationInfo(1.0f, 10)}),
                               make("QuantizationInfoInput", {QuantizationInfo(4.0f, 23)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u16);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8_SIGNED,
                       NEQuantizationLayerQASYMM8_SIGNEDGenFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(QuantizationSmallShapes,
                               make("DataTypeIn", DataType::QASYMM8_SIGNED),
                               make("DataTypeOut", {DataType::QASYMM8_SIGNED}),
                               make("QuantizationInfoOutput", {QuantizationInfo(1.0f, 10)}),
                               make("QuantizationInfoInput", {QuantizationInfo(2.0f, -5), QuantizationInfo(1.0f, 43)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}
FIXTURE_DATA_TEST_CASE(
    RunSmallQASYMM8,
    NEQuantizationLayerQASYMM8GenFixture<int8_t>,
    framework::DatasetMode::ALL,
    combine(QuantizationSmallShapes,
            make("DataType", DataType::QASYMM8_SIGNED),
            make("DataTypeOut", {DataType::QASYMM8}),
            make("QuantizationInfoOutput", {QuantizationInfo(2.0f, 10), QuantizationInfo(2.0f, -25)}),
            make("QuantizationInfoInput", {QuantizationInfo(1.0f, 30), QuantizationInfo(2.0f, -128)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(ConvertInt8toUint8,
                       NEQuantizationLayerQASYMM8_SIGNEDGenFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(QuantizationSmallShapes,
                               make("DataTypeIn", DataType::QASYMM8_SIGNED),
                               make("DataTypeOut", {DataType::QASYMM8}),
                               make("QuantizationInfoOutput", {QuantizationInfo(1.0f, 0)}),
                               make("QuantizationInfoInput", {QuantizationInfo(1.0f, -128)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // QuantizationLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
