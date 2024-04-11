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
#include "arm_compute/runtime/CL/functions/CLQuantizationLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/QuantizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float>    tolerance_f32(1.0f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr AbsoluteTolerance<uint8_t>  tolerance_u8(1);     /**< Tolerance value for comparing reference's output against implementation's output for QASYMM8 data types */
constexpr AbsoluteTolerance<int8_t>   tolerance_s8(1);     /**< Tolerance value for comparing reference's output against implementation's output for QASYMM8_SIGNED data types */
constexpr AbsoluteTolerance<uint16_t> tolerance_u16(1);    /**< Tolerance value for comparing reference's output against implementation's output for QASYMM16 data types */
const auto                            QuantizationSmallShapes = concat(datasets::Small3DShapes(), datasets::Small4DShapes());
const auto                            QuantizationLargeShapes = concat(datasets::Large3DShapes(), datasets::Large4DShapes());
} // namespace

TEST_SUITE(CL)
TEST_SUITE(QuantizationLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Wrong output data type
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32), // Wrong output data type
                                                       TensorInfo(TensorShape(16U, 16U, 2U, 5U), 1, DataType::F32),   // Mismatching shapes
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32), // Valid
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::U16),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),
                                                     })),
               framework::dataset::make("Expected", { false, false, false, true})),
               input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLQuantizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLQuantizationLayerQASYMM8Fixture = QuantizationValidationFixture<CLTensor, CLAccessor, CLQuantizationLayer, T, uint8_t>;
template <typename T>
using CLQuantizationLayerQASYMM8_SIGNEDFixture = QuantizationValidationFixture<CLTensor, CLAccessor, CLQuantizationLayer, T, int8_t>;
template <typename T>
using CLQuantizationLayerQASYMM16Fixture = QuantizationValidationFixture<CLTensor, CLAccessor, CLQuantizationLayer, T, uint16_t>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8, CLQuantizationLayerQASYMM8Fixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataTypeIn", DataType::F32)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8 })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8_SIGNED, CLQuantizationLayerQASYMM8_SIGNEDFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataTypeIn", DataType::F32)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8_SIGNED })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM16, CLQuantizationLayerQASYMM16Fixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataTypeIn", DataType::F32)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM16 })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_u16);
}
FIXTURE_DATA_TEST_CASE(RunLargeQASYMM8, CLQuantizationLayerQASYMM8Fixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(QuantizationLargeShapes,
                       framework::dataset::make("DataTypeIn", DataType::F32)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8 })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLargeQASYMM16, CLQuantizationLayerQASYMM16Fixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(QuantizationLargeShapes,
                       framework::dataset::make("DataTypeIn", DataType::F32)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM16 })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_u16);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8, CLQuantizationLayerQASYMM8Fixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataTypeIn", DataType::F16)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8 })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLargeQASYMM8, CLQuantizationLayerQASYMM8Fixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(QuantizationLargeShapes,
                       framework::dataset::make("DataTypeIn", DataType::F16)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8 })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
template <typename T>
using CLQuantizationLayerQASYMM8GenFixture = QuantizationValidationGenericFixture<CLTensor, CLAccessor, CLQuantizationLayer, T, uint8_t>;
template <typename T>
using CLQuantizationLayerQASYMM8_SIGNEDGenFixture = QuantizationValidationGenericFixture<CLTensor, CLAccessor, CLQuantizationLayer, T, int8_t>;
template <typename T>
using CLQuantizationLayerQASYMM16GenFixture = QuantizationValidationGenericFixture<CLTensor, CLAccessor, CLQuantizationLayer, T, uint16_t>;
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8, CLQuantizationLayerQASYMM8GenFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8 })),
                       framework::dataset::make("QuantizationInfoOutput", { QuantizationInfo(0.5f, 10) })),
                       framework::dataset::make("QuantizationInfoInput", { QuantizationInfo(2.0f, 15) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8_SIGNED, CLQuantizationLayerQASYMM8_SIGNEDGenFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataTypeIn", DataType::QASYMM8)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8_SIGNED })),
                       framework::dataset::make("QuantizationInfoOutput", { QuantizationInfo(1.0f, 10) })),
                       framework::dataset::make("QuantizationInfoInput", { QuantizationInfo(1.0f, 15) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_s8);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM16, CLQuantizationLayerQASYMM16GenFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataTypeIn", DataType::QASYMM8)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM16 })),
                       framework::dataset::make("QuantizationInfoOutput", { QuantizationInfo(1.0f, 10) })),
                       framework::dataset::make("QuantizationInfoInput", { QuantizationInfo(4.0f, 23) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_u16);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8_SIGNED, CLQuantizationLayerQASYMM8_SIGNEDGenFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataTypeIn", DataType::QASYMM8_SIGNED)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8_SIGNED })),
                       framework::dataset::make("QuantizationInfoOutput", { QuantizationInfo(1.0f, 10) })),
                       framework::dataset::make("QuantizationInfoInput", { QuantizationInfo(2.0f, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_s8);
}
FIXTURE_DATA_TEST_CASE(RunSmallQASYMM8, CLQuantizationLayerQASYMM8GenFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(QuantizationSmallShapes,
                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                       framework::dataset::make("DataTypeOut", { DataType::QASYMM8 })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.0f, 10) })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.0f, 30) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_u8);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // QuantizationLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
