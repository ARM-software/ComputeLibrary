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
RelativeTolerance<float> rel_tolerance_f32(0.0001f);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
AbsoluteTolerance<float> tolerance_f16(0.2f);
RelativeTolerance<float> rel_tolerance_f16(0.1f);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/** Tolerance for quantized operations */
RelativeTolerance<float> tolerance_quantized(1.f);

const auto ReductionOperations = framework::dataset::make("ReductionOperation",
{
    ReductionOperation::SUM,
    ReductionOperation::PROD,
    ReductionOperation::MIN,
    ReductionOperation::MAX,
});

const auto QuantizationInfos = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(1.f / 117, 10), // Numbers chosen so that the quantized values are in range of qasymm8_signed data type
    QuantizationInfo(1.f / 64, 5),
    QuantizationInfo(1.f / 32, 2)
});

const auto Axises = framework::dataset::make("Axis",
{ 0, 1, 2, 3 });

const auto KeepDims = framework::dataset::make("KeepDims", { true, false });

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ReductionOperation)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
    framework::dataset::make("InputInfo",          { TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Mismatching data type input/output
                                                     TensorInfo(TensorShape(128U, 64U), 2, DataType::F32), // Number of Input channels != 1
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::S16), // DataType != F32
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32), // Axis >= num_max_dimensions
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(128U, 64U), 1, DataType::F32) // Kept dimension when keep_dims = false
                                                   }),
    framework::dataset::make("OutputInfo",         { TensorInfo(TensorShape(1U, 64U), 1, DataType::F16),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::S16),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(1U, 64U), 1, DataType::F32)
                                                   })),
    framework::dataset::make("Axis",               { 0U, 0U, 0U, static_cast<unsigned int>(TensorShape::num_max_dimensions), 0U, 0U })),
    framework::dataset::make("KeepDims",           { true, true, true, true, true, false})),
    framework::dataset::make("Expected",           { false, false, false, false, true, false })),
    input_info, output_info, axis, keep_dims, expected)
{
    bool is_valid = bool(NEReductionOperation::validate(&input_info.clone()->set_is_resizable(false),
                                                        &output_info.clone()->set_is_resizable(true),
                                                        axis,
                                                        ReductionOperation::SUM_SQUARE,
                                                        keep_dims));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(ValidateNoPadding, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F32)), framework::dataset::make("Axis",
{ 0, 1 })), framework::dataset::make("ReductionOperation", {ReductionOperation::SUM,})), KeepDims),
               shape, data_type, axis, op, keep_dims)
{
    TensorShape         input_shape = TensorShape(shape);
    TensorInfo input_info   = TensorInfo(input_shape, 1, data_type);
    const bool is_arg_min_max = (op == ReductionOperation::ARG_IDX_MAX) || (op == ReductionOperation::ARG_IDX_MIN);
    const bool _keep_dims = keep_dims && !is_arg_min_max;
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_reduced_shape(shape, axis, keep_dims);

    // Create tensors
    Tensor src     = create_tensor<Tensor>(input_shape, data_type, 1, QuantizationInfo());
    Tensor dst     = create_tensor<Tensor>(output_shape, data_type, 1, QuantizationInfo());

    // Create and configure function
    NEReductionOperation reduction;
    reduction.configure(&src, &dst, axis, op, _keep_dims);

    validate(src.info()->padding(), PaddingSize(0, 0, 0, 0));
    validate(dst.info()->padding(), PaddingSize(0, 0, 0, 0));
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEReductionOperationFixture = ReductionOperationFixture<Tensor, Accessor, NEReductionOperation, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReductionOperationFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F32)), Axises), ReductionOperations), KeepDims))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEReductionOperationFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::F32)), Axises), ReductionOperations), KeepDims))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReductionOperationFixture<half>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F16)), Axises), ReductionOperations), KeepDims))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEReductionOperationFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::F16)), Axises), ReductionOperations), KeepDims))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, 0, tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename T>
using NEReductionOperationQuantizedFixture = ReductionOperationQuantizedFixture<Tensor, Accessor, NEReductionOperation, T>;

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReductionOperationQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), Axises),
                                               ReductionOperations),
                                       QuantizationInfos),
                               KeepDims))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReductionOperationQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)), Axises),
                                               ReductionOperations),
                                       QuantizationInfos),
                               KeepDims))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // ReductionOperation
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
