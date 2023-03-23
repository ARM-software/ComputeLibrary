/*
 * Copyright (c) 2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEMatMul.h"

#include "tests/NEON/Accessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

#include "tests/datasets/LargeMatMulDataset.h"
#include "tests/datasets/SmallMatMulDataset.h"
#include "tests/validation/fixtures/MatMulFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(MatMul)

constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for FP32 data types */
const AbsoluteTolerance<half>      tolerance_fp16(half(0.1f));

// clang-format off
// *INDENT-OFF*
// Validation Tests
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(9U, 6U), 1, DataType::F32),        // Mismatching datatype
                                             TensorInfo(TensorShape(9U, 6U), 1, DataType::S32),        // Unsupported datatypes
                                             TensorInfo(TensorShape(9U, 6U, 2U), 1, DataType::F32),    // Broadcasting in batch dimension not supported
                                             TensorInfo(TensorShape(9U, 6U), 1, DataType::F32),        // Invalid shape for multiplication
                                             TensorInfo(TensorShape(9U, 6U), 1, DataType::F32),
                                             TensorInfo(TensorShape(9U, 6U , 12U) , 1 , DataType::F32),
                                             TensorInfo(TensorShape(9U, 6U , 12U) , 1 , DataType::F32), // Tensors are not dynamic
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(5U, 9U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(5U, 9U), 1, DataType::S32),
                                            TensorInfo(TensorShape(5U, 9U, 1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 12U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 9U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 9U, 12U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 9U, 12U), 1, DataType::F32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(5U, 6U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 6U), 1, DataType::S32),
                                            TensorInfo(TensorShape(5U, 6U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 6U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 6U), 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 6U, 12U) , 1, DataType::F32),
                                            TensorInfo(TensorShape(5U, 6U, 12U) , 1, DataType::F32),
                                           })),
    framework::dataset::make( "TensorIsConst", {false, false, false, false, false , false, true} )),
    framework::dataset::make("Expected", { false, false, false, false, true, true, false })),
    a_info, b_info, output_info, are_tensors_const, expected)
{
    TensorInfo a{a_info};
    TensorInfo b{b_info};
    a.set_are_values_constant(are_tensors_const);
    b.set_are_values_constant(are_tensors_const);
    Status status =  NEMatMul::validate(&a,
                                        &b,
                                        &output_info,
                                        MatMulInfo(),
                                        CpuMatMulSettings());
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// *INDENT-ON*
// clang-format on

// Generic Template
template <typename T>
using NEMatMulFixture = MatMulValidationWithActivationFixture<Tensor, Accessor, NEMatMul, CpuMatMulSettings, T>;

// Fast math Template
template <typename T>
using NEMatMulFastMathFixture = MatMulGenericValidationFixture<Tensor, Accessor, NEMatMul, CpuMatMulSettings, T>;

template <typename T>
using NEMatMulDynamicTensorsFixture = MatMulValidationWithDynamicTensorsFixture<Tensor, Accessor, NEMatMul, CpuMatMulSettings, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEMatMulFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallMatMulDataset(),
                                                                                                                    framework::dataset::make("TransposeA", { false, true })),
                                                                                                                    framework::dataset::make("TransposeB", { false, true })),
                                                                                                            framework::dataset::make("DataType", DataType::F32)),
                                                                                                    framework::dataset::make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEMatMulFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeMatMulDataset(),
                                                                                                                  framework::dataset::make("TransposeA", { false, true })),
                                                                                                                  framework::dataset::make("TransposeB", { false, true })),
                                                                                                          framework::dataset::make("DataType", DataType::F32)),
                                                                                                  framework::dataset::make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunHighDimensions, NEMatMulFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::HighDimensionalMatMulDataset(),
                                                                                                                   framework::dataset::make("TransposeA", { false, true })),
                                                                                                                   framework::dataset::make("TransposeB", { false, true })),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                           framework::dataset::make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunStressDynamicTensors, NEMatMulDynamicTensorsFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallMatMulDataset(),
                       framework::dataset::make("TransposeA", { false, true })),
                       framework::dataset::make("TransposeB", { false, true })),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })),
                       framework::dataset::make("NumberOfRuns", 5)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_BF16
/* Note : MatMul BF16 is enabled by specifying FP32 datatype and enabling the fast math setting */
constexpr AbsoluteTolerance<float> tolerance_bf16(0.001f);
TEST_SUITE(BF16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEMatMulFastMathFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(datasets::SmallMatMulDataset(),
                                                                                                                    framework::dataset::make("TransposeA", { false, true })),
                                                                                                                    framework::dataset::make("TransposeB", { false, true })),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    framework::dataset::make("ActivationInfo", { ActivationLayerInfo() })),
                                                                                                                    framework::dataset::make("RunTimes", { 0 })),
                                                                                                            framework::dataset::make("Settings", { CpuMatMulSettings().fast_math(true) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_bf16);
}
TEST_SUITE_END() // BF16
#endif           /* ARM_COMPUTE_ENABLE_BF16 */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEMatMulFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallMatMulDataset(),
                                                                                                                   framework::dataset::make("TransposeA", { false, true })),
                                                                                                                   framework::dataset::make("TransposeB", { false, true })),
                                                                                                           framework::dataset::make("DataType", DataType::F16)),
                                                                                                   framework::dataset::make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEMatMulFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeMatMulDataset(),
                                                                                                                 framework::dataset::make("TransposeA", { false, true })),
                                                                                                                 framework::dataset::make("TransposeB", { false, true })),
                                                                                                         framework::dataset::make("DataType", DataType::F16)),
                                                                                                 framework::dataset::make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
FIXTURE_DATA_TEST_CASE(RunStressDynamicTensors, NEMatMulDynamicTensorsFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallMatMulDataset(),
                       framework::dataset::make("TransposeA", { false, true })),
                       framework::dataset::make("TransposeB", { false, true })),
                       framework::dataset::make("DataType", DataType::F16)),
                       framework::dataset::make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })),
                       framework::dataset::make("NumberOfRuns", 5)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE_END() // Float

TEST_SUITE_END() // MatMul
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
