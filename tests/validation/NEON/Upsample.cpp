/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEUpsampleLayer.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/UpsampleLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(UpsampleLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Mismatching data type
                                            TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Invalid output shape
                                            TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Invalid stride
                                            TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Invalid policy
                                            TensorInfo(TensorShape(32U, 32U), 1, DataType::F32),
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(20U, 20U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(20U, 10U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(20U, 20U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(20U, 20U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(64U, 64U), 1, DataType::F32),
                                          })),
    framework::dataset::make("PadInfo", { Size2D(2, 2),
                                          Size2D(2, 2),
                                          Size2D(1, 1),
                                          Size2D(2, 2),
                                          Size2D(2, 2),
                                           })),
   framework::dataset::make("UpsamplingPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR,
                                                  InterpolationPolicy::NEAREST_NEIGHBOR,
                                                  InterpolationPolicy::NEAREST_NEIGHBOR,
                                                  InterpolationPolicy::BILINEAR,
                                                  InterpolationPolicy::NEAREST_NEIGHBOR,
                                                })),
    framework::dataset::make("Expected", { false, false, false, false, true })),
    input_info, output_info, pad_info, policy, expected)
{
    bool is_valid = bool(NEUpsampleLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pad_info, policy));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEUpsampleLayerFixture = UpsampleLayerFixture<Tensor, Accessor, NEUpsampleLayer, T>;

template <typename T>
using NEUpsampleLayerQuantizedFixture = UpsampleLayerQuantizedFixture<Tensor, Accessor, NEUpsampleLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEUpsampleLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("PadInfo", { Size2D(2, 2) })),
                                                                                                           framework::dataset::make("UpsamplingPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEUpsampleLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::F16)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                  framework::dataset::make("PadInfo", { Size2D(2, 2) })),
                                                                                                          framework::dataset::make("UpsamplingPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEUpsampleLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                      framework::dataset::make("PadInfo", { Size2D(2, 2) })),
                                                                                                                      framework::dataset::make("UpsamplingPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR })),
                                                                                                                      framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255.f, 10))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEUpsampleLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                     framework::dataset::make("PadInfo", { Size2D(2, 2) })),
                                                                                                                     framework::dataset::make("UpsamplingPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR })),
                                                                                                                     framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255.f, 10))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // UpsampleLayer
TEST_SUITE_END() // NEON

} // namespace validation
} // namespace test
} // namespace arm_compute
