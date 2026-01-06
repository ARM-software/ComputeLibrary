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
#include "arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/L2NormalizeLayerFixture.h"
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
/** Tolerance for float operations */
RelativeTolerance<float> tolerance_f32(0.00001f);
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<float> tolerance_f16(0.2f);
#endif // ARM_COMPUTE_ENABLE_FP16
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(L2NormalizeLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo",  { TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Mismatching data type input/output
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Mismatching shape input/output
                                             TensorInfo(TensorShape(128U, 64U), 2, DataType::F32), // Number of Input channels != 1
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::S16), // DataType != F32
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32)
                                           }),
    make("OutputInfo", { TensorInfo(TensorShape(128U, 64U), 1, DataType::F16),
                                             TensorInfo(TensorShape(256U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::S16),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                             TensorInfo(TensorShape(128U, 64U), 1, DataType::F32)
                                           }),
    make("Axis",       {
                                            0,
                                            0,
                                            0,
                                            0,
                                            static_cast<int>(TensorShape::num_max_dimensions),
                                            3,
                                            -2,
                                            0 }),
    make("Expected",   { false, false, false, false, true, true, true, true })
    ),
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
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEL2NormalizeLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC}),
                               make("Axis", {-1, 0, 1, 2}),
                               make("Epsilon", {1e-6})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEL2NormalizeLayerFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(),
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC}),
                               make("Axis", {-1, 0, 2}),
                               make("Epsilon", {1e-6})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEL2NormalizeLayerFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::F16),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC}),
                               make("Axis", {-1, 0, 1, 2}),
                               make("Epsilon", {1e-6})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEL2NormalizeLayerFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(),
                               make("DataType", DataType::F16),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC}),
                               make("Axis", {-1, 0, 2}),
                               make("Epsilon", {1e-6})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE_END() // L2NormalizeLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
