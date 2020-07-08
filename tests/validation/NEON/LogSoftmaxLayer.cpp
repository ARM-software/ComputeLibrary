/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
constexpr RelativeTolerance<float> tolerance_f32(0.00001f);
RelativeTolerance<half>            tolerance_f16(half(0.2));

/** Tolerance for quantized operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(LogSoftmaxLayer)

template <typename T>
using NELogSoftmaxLayerFixture = SoftmaxValidationFixture<Tensor, Accessor, NELogSoftmaxLayer, T, true>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NELogSoftmaxLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::F16)),
                                                                                                                    framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                            framework::dataset::make("ReduceEndAxis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NELogSoftmaxLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                      framework::dataset::make("DataType", DataType::F16)),
                                                                                                                      framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                              framework::dataset::make("ReduceEndAxis", { 0, 1, 2 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NELogSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                  framework::dataset::make("DataType", DataType::F16)),
                                                                                                                  framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                          framework::dataset::make("ReduceEndAxis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() //FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall2D, NELogSoftmaxLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::F32)),
                                                                                                                       framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                               framework::dataset::make("ReduceEndAxis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NELogSoftmaxLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::F32)),
                                                                                                                       framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                               framework::dataset::make("ReduceEndAxis", { 0, 1, 2 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NELogSoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                           framework::dataset::make("ReduceEndAxis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() //FP32
TEST_SUITE_END() //Float

template <typename T>
using NELogSoftmaxLayerQuantizedFixture = SoftmaxValidationQuantizedFixture<Tensor, Accessor, NELogSoftmaxLayer, T, true>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall2D, NELogSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                    combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                            framework::dataset::make("Beta", { 1.0f, 2.f }))),
                                                                                                                    framework::dataset::make("ReduceEndAxis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NELogSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                    combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                            framework::dataset::make("Beta", { 1.0f, 2.f }))),
                                                                                                                    framework::dataset::make("ReduceEndAxis", { 0, 1, 2 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NELogSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                      combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                              framework::dataset::make("Beta", { 1.0f, 2.0f }))),
                                                                                                                      framework::dataset::make("ReduceEndAxis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() //QASYMM8
TEST_SUITE_END() //Quantized

TEST_SUITE_END() //LogSoftmaxLayer
TEST_SUITE_END() //NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
