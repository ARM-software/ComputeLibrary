/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "tests/CL/CLAccessor.h"
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
AbsoluteTolerance<float> tolerance_f32(0.001f);
RelativeTolerance<float> rel_tolerance_f32(0.00001f);
AbsoluteTolerance<float> tolerance_f16(0.5f);
RelativeTolerance<float> rel_tolerance_f16(0.2f);
/** Tolerance for quantized operations */
RelativeTolerance<float> tolerance_qasymm8(1);

const auto ReductionOperationsSumProdMean = framework::dataset::make("ReductionOperationsSumProdMean",
{
    ReductionOperation::SUM,
    ReductionOperation::PROD,
    ReductionOperation::MEAN_SUM

});
const auto ReductionOperationsMinMax = framework::dataset::make("ReductionMinMax",
{
    ReductionOperation::MIN,
    ReductionOperation::MAX,
});

const auto KeepDimensions = framework::dataset::make("KeepDims", { true, false });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ReductionOperation)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
    framework::dataset::make("InputInfo",          { TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Mismatching data type input/output
                                                     TensorInfo(TensorShape(128U, 64U), 3, DataType::F32), // Number of Input channels != 1
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::S16), // DataType != QASYMM8/F16/F32
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Axis >= num_max_dimensions
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::QASYMM8), // Axis == 0 and SUM_SQUARE and QASYMM8
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32) // Kept Dimension when keep_dims = false

                                                   }),
    framework::dataset::make("OutputInfo",         { TensorInfo(TensorShape(1U, 64U), 1, DataType::F16),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::S16),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::QASYMM8),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32)
                                                   })),
    framework::dataset::make("Axis",               { 0U, 0U, 0U, static_cast<unsigned int>(TensorShape::num_max_dimensions), 1U, 0U, 0U })),
    framework::dataset::make("KeepDims",           { true, true, true, true, true, true, false })),
    framework::dataset::make("Expected",           { false, false, false, false, false, true , false })),
    input_info, output_info, axis, keep_dims, expected)
{
    bool is_valid = bool(CLReductionOperation::validate(&input_info.clone()->set_is_resizable(false),
                                                        &output_info.clone()->set_is_resizable(true),
                                                        axis,
                                                        ReductionOperation::SUM_SQUARE,
                                                        keep_dims));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLReductionOperationFixture = ReductionOperationFixture<CLTensor, CLAccessor, CLReductionOperation, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall4D, CLReductionOperationFixture<half>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F16)), framework::dataset::make("Axis", { 0, 1, 2, 3 })),
                                       concat(ReductionOperationsSumProdMean,
                                              ReductionOperationsMinMax)),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLReductionOperationFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F16)), framework::dataset::make("Axis", { 0, 1, 2, 3 })), concat(ReductionOperationsSumProdMean,
                                       ReductionOperationsMinMax)),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0, tolerance_f16);
}
TEST_SUITE_END() // F16
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall4D, CLReductionOperationFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F32)), framework::dataset::make("Axis", { 0, 1, 2, 3 })),
                                       concat(ReductionOperationsSumProdMean,
                                              ReductionOperationsMinMax)),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLReductionOperationFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F32)), framework::dataset::make("Axis", { 0, 1, 2, 3 })), concat(ReductionOperationsSumProdMean,
                                       ReductionOperationsMinMax)),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0, tolerance_f32);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

template <typename T>
using CLReductionOperationQuantizedFixture = ReductionOperationQuantizedFixture<CLTensor, CLAccessor, CLReductionOperation, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReductionOperationQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), framework::dataset::make("Axis", { 0, 1, 2, 3 })),
                                               ReductionOperationsSumProdMean),
                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(1.f / 64, 2))),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmallMinMax, CLReductionOperationQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), framework::dataset::make("Axis", { 0, 1, 2, 3 })),
                                               ReductionOperationsMinMax),
                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(1.f / 64, 2))),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReductionOperationQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)), framework::dataset::make("Axis", { 0, 1, 2, 3 })),
                                               ReductionOperationsSumProdMean),
                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(1.f / 64, 2))),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmallMinMax, CLReductionOperationQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)), framework::dataset::make("Axis", { 0, 1, 2, 3 })),
                                               ReductionOperationsMinMax),
                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(1.f / 64, 2))),
                               KeepDimensions))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // Reduction
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
