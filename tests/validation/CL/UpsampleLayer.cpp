/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLUpsampleLayer.h"
#include "tests/CL/CLAccessor.h"
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
namespace
{
constexpr AbsoluteTolerance<float> tolerance(0.001f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(UpsampleLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Mismatching data type
                                            TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Invalid output shape
                                            TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Invalid stride
                                            TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32), // Invalid policy
                                            TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32),
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(20U, 20U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(20U, 10U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(20U, 20U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(20U, 20U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(20U, 20U, 2U), 1, DataType::F32),
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
    input_info, output_info, pad_info, upsampling_policy, expected)
{
    bool is_valid = bool(CLUpsampleLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pad_info, upsampling_policy));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLUpsampleLayerFixture = UpsampleLayerFixture<CLTensor, CLAccessor, CLUpsampleLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLUpsampleLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("PadInfo", { Size2D(2, 2) })),
                                                                                                           framework::dataset::make("UpsamplingPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLUpsampleLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::F16)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                  framework::dataset::make("PadInfo", { Size2D(2, 2) })),
                                                                                                          framework::dataset::make("UpsamplingPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE_END() // UpsampleLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
