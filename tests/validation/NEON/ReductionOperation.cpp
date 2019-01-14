/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReductionOperationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
AbsoluteTolerance<float> tolerance_f32(0.0001f);
RelativeTolerance<float> rel_tolerance_f32(0.00001f);
/** Tolerance for quantized operations */
RelativeTolerance<float> tolerance_qasymm8(1);

const auto ReductionOperations = framework::dataset::make("ReductionOperation",
{
    ReductionOperation::SUM,
    ReductionOperation::PROD
});

const auto QuantizationInfos = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(1.f / 128, -10),
    QuantizationInfo(1.f / 64, -5),
    QuantizationInfo(1.f / 32, -2)
});

const auto Axises = framework::dataset::make("Axis",
{ 0, 1, 2, 3 });

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ReductionOperation)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
    framework::dataset::make("InputInfo",          { TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Mismatching data type input/output
                                                     TensorInfo(TensorShape(128U, 64U), 2, DataType::F32), // Number of Input channels != 1
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::S16), // DataType != F32
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Axis >= num_max_dimensions
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32)
                                                   }),
    framework::dataset::make("OutputInfo",         { TensorInfo(TensorShape(1U, 64U), 1, DataType::F16),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::S16),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32)
                                                   })),
    framework::dataset::make("Axis",               { 0U, 0U, 0U, static_cast<unsigned int>(TensorShape::num_max_dimensions), 0U })),
    framework::dataset::make("Expected",           { false, false, false, false, true })),
    input_info, output_info, axis, expected)
{
    bool is_valid = bool(NEReductionOperation::validate(&input_info.clone()->set_is_resizable(false),
                                                        &output_info.clone()->set_is_resizable(true),
                                                        axis,
                                                        ReductionOperation::SUM_SQUARE));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEReductionOperationFixture = ReductionOperationFixture<Tensor, Accessor, NEReductionOperation, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReductionOperationFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F32)), Axises), ReductionOperations))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEReductionOperationFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::F32)), Axises), ReductionOperations))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, tolerance_f32);
}
TEST_SUITE_END() // FP32

template <typename T>
using NEReductionOperationQuantizedFixture = ReductionOperationQuantizedFixture<Tensor, Accessor, NEReductionOperation, T>;

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReductionOperationQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), Axises),
                                       ReductionOperations),
                               QuantizationInfos))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEReductionOperationQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), Axises),
                                       ReductionOperations),
                               QuantizationInfos))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE_END() // ReductionOperation
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
