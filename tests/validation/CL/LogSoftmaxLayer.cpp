/*
 * Copyright (c) 2019-2020, 2024-2025 Arm Limited.
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
using framework::dataset::make;
namespace
{
/** Tolerance for float operations */
RelativeTolerance<half>  tolerance_f16(half(0.2));
RelativeTolerance<float> tolerance_f32(0.001f);

/** Tolerance for quantized operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1U);
constexpr AbsoluteTolerance<int8_t> tolerance_qasymm8_signed(1);

} // namespace


TEST_SUITE(CL)
TEST_SUITE(LogSoftmaxLayer)

template <typename T>
using CLLogSoftmaxLayerFixture = SoftmaxValidationFixture<CLTensor, CLAccessor, CLLogSoftmaxLayer, T, true>;

template <typename T>
using CLLogSoftmaxLayerQuantizedFixture = SoftmaxValidationQuantizedFixture<CLTensor, CLAccessor, CLLogSoftmaxLayer, T, true>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLLogSoftmaxLayerFixture<half>, framework::DatasetMode::ALL, combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                      make("DataType", DataType::F16),
                                                                                                              make("Beta", { 1.0f, 2.0f }),
                                                                                                      make("Axis", { 0, -1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLLogSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                  make("DataType", DataType::F16),
                                                                                                                  make("Beta", { 1.0f, 2.0f }),
                                                                                                          make("Axis", { 0 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(Run4D, CLLogSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::SoftmaxLayer4DShapes(),
                                                                                                                       make("DataType", DataType::F16),
                                                                                                               make("Beta", { 1.0f, 2.0f }),
                                                                                                       make("Axis", { 0, -3, 2 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLLogSoftmaxLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                       make("DataType", DataType::F32),
                                                                                                               make("Beta", { 1.0f, 2.0f }),
                                                                                                       make("Axis", { 0, 1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLLogSoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                   make("DataType", DataType::F32),
                                                                                                                   make("Beta", { 1.0f, 2.0f }),
                                                                                                           make("Axis", { 0 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(Run4D, CLLogSoftmaxLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SoftmaxLayer4DShapes(),
                                                                                                                    make("DataType", DataType::F32),
                                                                                                            make("Beta", { 1.0f, 2.0f }),
                                                                                                    make("Axis", { 0, -4, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLLogSoftmaxLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
    combine(datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::QASYMM8_SIGNED),
        make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, 1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLLogSoftmaxLayerQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY,
    combine(datasets::SoftmaxLayerLargeShapes(),
        make("DataType", DataType::QASYMM8_SIGNED),
        make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE(Run4D, CLLogSoftmaxLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
    combine(datasets::SoftmaxLayer4DShapes(),
        make("DataType", DataType::QASYMM8_SIGNED),
        make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -4, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLLogSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
    combine(datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::QASYMM8),
        make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, 1 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLLogSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
    combine(datasets::SoftmaxLayerLargeShapes(),
        make("DataType", DataType::QASYMM8),
        make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(Run4D, CLLogSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
    combine(datasets::SoftmaxLayer4DShapes(),
        make("DataType", DataType::QASYMM8),
        make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -4, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // LogSoftmaxLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
