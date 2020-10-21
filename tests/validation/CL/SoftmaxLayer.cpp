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
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SoftmaxLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
RelativeTolerance<half>  tolerance_f16(half(0.2));
RelativeTolerance<float> tolerance_f32(0.001f);

/** Tolerance for quantized operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_signed(1);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::QASYMM8,
    DataType::F16,
    DataType::F32,
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(SoftmaxLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8, // Invalid output quantization info
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Window shrink
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),// Invalid input dimensionality
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8_SIGNED,
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8_SIGNED, // Invalid axis high
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8_SIGNED, // Invalid axis low
                                                                  QuantizationInfo(1.f/256, 12))
                                                      }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 0)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8_SIGNED,
                                                                  QuantizationInfo(1.f/256, -128)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8_SIGNED,
                                                                  QuantizationInfo(1.f/256, -128)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8_SIGNED,
                                                                  QuantizationInfo(1.f/256, -128)),
                                                     })),
               framework::dataset::make("beta", { 1.0,
                                                  2.0,
                                                  1.0,
                                                  2.0,
                                                  1.0,
                                                  2.0,
                                                  1.0,
                                                  2.0,
                                                  1.0,
                                                  2.0,
                                                })),
               framework::dataset::make("axis", {
                                                  0,
                                                  0,
                                                  0,
                                                  0,
                                                  0,
                                                  1,
                                                  0,
                                                  -1,
                                                  2,
                                                  -3,
                                                })),
               framework::dataset::make("Expected", { false, false, false, false, false, true, true, true, false, false })),
               input_info, output_info, beta, axis, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLSoftmaxLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), beta, axis)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLSoftmaxLayerFixture = SoftmaxValidationFixture<CLTensor, CLAccessor, CLSoftmaxLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                   framework::dataset::make("Axis", { 0, -1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::F16)),
                                                                                                               framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                       framework::dataset::make("Axis", { 0 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(Run4D, CLSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayer4DShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::F16)),
                                                                                                            framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                    framework::dataset::make("Axis", { 0, -1, 2 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                            framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                    framework::dataset::make("Axis", { 0, 1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                                                                                framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                        framework::dataset::make("Axis", { 0 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(Run4D, CLSoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayer4DShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::F32)),
                                                                                                             framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                     framework::dataset::make("Axis", { 0, -2, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLSoftmaxLayerQuantizedFixture = SoftmaxValidationQuantizedFixture<CLTensor, CLAccessor, CLSoftmaxLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                       combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                               framework::dataset::make("Beta", { 1.0f, 2.f }))),
                                                                                                               framework::dataset::make("Axis", { 0, 1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                   combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f }))),
                                                                                                                   framework::dataset::make("Axis", { 0 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(Run4D, CLSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayer4DShapes(),
                                                                                                                        framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                        combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                                framework::dataset::make("Beta", { 1.0f, 2.0f }))),
                                                                                                                framework::dataset::make("Axis", { 0, -4, 1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)

FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                      combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                              framework::dataset::make("Beta", { 1.0f, 2.f }))),
                                                                                                              framework::dataset::make("Axis", { 0, 1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_signed);
}

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // SoftmaxLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
