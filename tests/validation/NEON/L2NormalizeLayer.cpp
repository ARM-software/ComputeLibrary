/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/L2NormalizeLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
RelativeTolerance<float> tolerance_f32(0.00001f);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
RelativeTolerance<float> tolerance_f16(0.2f);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(L2NormalizeLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
    framework::dataset::make("InputInfo",  { TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Mismatching data type input/output
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Mismatching shape input/output
                                             TensorInfo(TensorShape(128U, 64U), 2, DataType::F32), // Number of Input channels != 1
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::S16), // DataType != F32
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Axis >= num_max_dimensions
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Axis > 2
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32)
                                           }),
    framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(128U, 64U), 1, DataType::F16),
                                             TensorInfo(TensorShape(256U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::S16),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32)
                                           })),
    framework::dataset::make("Axis",       { 0U, 0U, 0U, 0U, static_cast<unsigned int>(TensorShape::num_max_dimensions), 3U, 0U })),
    framework::dataset::make("Expected",   { false, false, false, false, false, false, true })),
    input_info, output_info, axis, expected)
{
    bool is_valid = bool(NEL2NormalizeLayer::validate(&input_info.clone()->set_is_resizable(false),
                                                      &output_info.clone()->set_is_resizable(false),
                                                      axis));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEL2NormalizeLayerFixture = L2NormalizeLayerValidationFixture<Tensor, Accessor, NEL2NormalizeLayer, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEL2NormalizeLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F32)), framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("Axis", { 0, 1, 2 })),
                               framework::dataset::make("Epsilon", { 1e-12 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEL2NormalizeLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F32)), framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("Axis", { 0, 1, 2 })),
                               framework::dataset::make("Epsilon", { 1e-12 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEL2NormalizeLayerFixture<half>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F16)), framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("Axis", { 0, 1, 2 })),
                               framework::dataset::make("Epsilon", { 1e-12 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEL2NormalizeLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F16)), framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("Axis", { 0, 1, 2 })),
                               framework::dataset::make("Epsilon", { 1e-12 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE_END() // L2NormalizeLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
