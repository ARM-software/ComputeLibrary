/*
 * Copyright (c) 2025-2026 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuMeanStdDevNormalization.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuMeanStdDevNormalizationFixture.h"
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
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<half> tolerance_f16(half(0.2f));
#endif /* ARM_COMPUTE_ENABLE_FP16 */
RelativeTolerance<float>   tolerance_f32(0.001f);
RelativeTolerance<uint8_t> tolerance_qasymm8(1);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuMeanStdDevNormalization)

DATA_TEST_CASE(Validate,
               framework::DatasetMode::ALL,
               zip(make("InputInfo",
                        {
                            TensorInfo(TensorShape(27U, 13U), 1, DataType::F32), // Mismatching data type input/output
                            TensorInfo(TensorShape(27U, 13U), 1, DataType::F32), // Mismatching shapes
                            TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        }),
                   make("OutputInfo",
                        {
                            TensorInfo(TensorShape(27U, 13U), 1, DataType::F16),
                            TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                            TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        }),
                   make("Expected", {false, false, true})),
               input_info,
               output_info,
               expected)
{
    ARM_COMPUTE_EXPECT(
        bool(experimental::op::CpuMeanStdDevNormalization::validate(
            &input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected,
        framework::LogLevel::ERRORS);
}

template <typename T>
using CpuMeanStdDevNormalizationFixture =
    CpuMeanStdDevNormalizationValidationFixture<Tensor, Accessor, experimental::op::CpuMeanStdDevNormalization, T>;

template <typename T>
using CpuMeanStdDevNormalizationFloatThreadSafeFixture =
    CpuMeanStdDevNormalizationFloatThreadSafeValidationFixture<Tensor,
                                                               Accessor,
                                                               experimental::op::CpuMeanStdDevNormalization,
                                                               T>;

template <typename T>
using CpuMeanStdDevNormalizationQuantizedThreadSafeFixture =
    CpuMeanStdDevNormalizationQuantizedThreadSafeValidationFixture<Tensor,
                                                                   Accessor,
                                                                   experimental::op::CpuMeanStdDevNormalization,
                                                                   T>;

TEST_SUITE(SmokeTest)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuMeanStdDevNormalizationFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Small2DShapes(),
                               make("InPlace", {false, true}),
                               make("Epsilon", {1e-7}),
                               make("DataType", DataType::F32)))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_f32);
    }
}
TEST_SUITE_END() // SmokeTest

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuMeanStdDevNormalizationFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small2DShapes(),
                               make("InPlace", {false, true}),
                               make("Epsilon", {1e-7}),
                               make("DataType", DataType::F32)))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_f32);
    }
}

TEST_SUITE_END() // F32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuMeanStdDevNormalizationFloatThreadSafeFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small2DShapes(),
                               make("InPlace", {false, true}),
                               make("Epsilon", {1e-7}),
                               make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], tolerance_f16);
        }
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // F16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE_END() // Float

TEST_SUITE(Quantized)

// Int8 and UInt8 are very similar, therefore no need to test both from thread-safety perspective
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuMeanStdDevNormalizationQuantizedThreadSafeFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small2DShapes(),
                               make("InPlace", {false, true}),
                               make("Epsilon", {1e-7}),
                               make("DataType", DataType::QASYMM8),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10)})))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_qasymm8);
    }
}

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // ThreadSafety
#endif           // #ifndef BARE_METAL

TEST_SUITE_END() // CpuMeanStdDevNormalization
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON

} // namespace validation
} // namespace test
} // namespace arm_compute
