/*
 * Copyright (c) 2018-2021, 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuElementwise.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuElementwiseFixture.h"
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
constexpr RelativeTolerance<float>   tolerance_div_fp32{0.000001f};
constexpr AbsoluteTolerance<uint8_t> abs_tolerance_qasymm8{1};

const auto ElementwiseFP32Dataset =
    combine(make("DataType", DataType::F32), make("DataType", DataType::F32), make("DataType", DataType::F32));

const auto ElementwiseFP16Dataset =
    combine(make("DataType", DataType::F16), make("DataType", DataType::F16), make("DataType", DataType::F16));

const auto ElementwiseS32Dataset =
    combine(make("DataType", DataType::S32), make("DataType", DataType::S32), make("DataType", DataType::S32));

const auto ElementwiseQuantizedDataset = combine(
    make("DataType", DataType::QASYMM8), make("DataType", DataType::QASYMM8), make("DataType", DataType::QASYMM8));

const auto ElementwiseQuantizedSignedDataset = combine(make("DataType", DataType::QASYMM8_SIGNED),
                                                       make("DataType", DataType::QASYMM8_SIGNED),
                                                       make("DataType", DataType::QASYMM8_SIGNED));

const auto ElementwiseQuantizationInfo = combine(make("QuantizationInfoIn1", {QuantizationInfo(0.5f, 10)}),
                                                 make("QuantizationInfoIn2", {QuantizationInfo(0.5f, 20)}),
                                                 make("QuantizationInfoOut", {QuantizationInfo(0.5f, 50)}));

const auto InPlaceDataSet    = make("InPlace", {false, true});
const auto OutOfPlaceDataSet = make("InPlace", {false});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)

TEST_SUITE(CpuElementwiseDivision)
template <typename T>
using CpuElementwiseDivisionFixture =
    CpuElementwiseDivisionValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseDivision, T>;

template <typename T>
using CpuElementwiseDivisionThreadSafeFixture =
    CpuElementwiseDivisionThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseDivision, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseDivisionFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
    }
}
TEST_SUITE_END() // F32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseDivisionFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP16Dataset, InPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
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

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Float)

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuElementwiseDivisionThreadSafeFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
    }
}
TEST_SUITE_END() // F32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseDivisionThreadSafeFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP16Dataset, InPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
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
TEST_SUITE_END() // ThreadSafety
#endif           // #ifndef BARE_METAL
TEST_SUITE_END() // CpuElementwiseDivision

TEST_SUITE(CpuElementwiseMax)
template <typename T>
using CpuElementwiseMaxFixture =
    CpuElementwiseMaxValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMax, T>;

template <typename T>
using CpuElementwiseMaxThreadSafeFixture =
    CpuElementwiseMaxThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMax, T>;

template <typename T>
using CpuElementwiseMaxQuantizedThreadSafeFixture =
    CpuElementwiseMaxQuantizedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMax, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMaxFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // F32
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMaxFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP16Dataset, InPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
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
#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuElementwiseMaxThreadSafeFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
    }
}
TEST_SUITE_END() // F32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMaxThreadSafeFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP16Dataset, InPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
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

TEST_SUITE(Integer)
TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuElementwiseMaxThreadSafeFixture<int32_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseS32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // S32
TEST_SUITE_END() // Integer

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(
    ConfigureOnceUseFromDifferentThreads,
    CpuElementwiseMaxQuantizedThreadSafeFixture<int8_t>,
    framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), ElementwiseQuantizedSignedDataset, ElementwiseQuantizationInfo, OutOfPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], abs_tolerance_qasymm8);
    }
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(
    ConfigureOnceUseFromDifferentThreads,
    CpuElementwiseMaxQuantizedThreadSafeFixture<uint8_t>,
    framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), ElementwiseQuantizedDataset, ElementwiseQuantizationInfo, OutOfPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], abs_tolerance_qasymm8);
    }
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ThreadSafety
#endif           // #ifndef BARE_METAL
TEST_SUITE_END() // CpuElementwiseMax

TEST_SUITE(CpuElementwiseMin)

template <typename T>
using CpuElementwiseMinFixture =
    CpuElementwiseMinValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMin, T>;

template <typename T>
using CpuElementwiseMinThreadSafeFixture =
    CpuElementwiseMinThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMin, T>;

template <typename T>
using CpuElementwiseMinQuantizedThreadSafeFixture =
    CpuElementwiseMinQuantizedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMin, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMinFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // F32
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMinFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP16Dataset, InPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
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
#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuElementwiseMinThreadSafeFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
    }
}
TEST_SUITE_END() // F32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMinThreadSafeFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP16Dataset, InPlaceDataSet))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], tolerance_div_fp32, 0.01);
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
TEST_SUITE(Integer)
TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuElementwiseMinThreadSafeFixture<int32_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseS32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // S32
TEST_SUITE_END() // Integer
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(
    ConfigureOnceUseFromDifferentThreads,
    CpuElementwiseMinQuantizedThreadSafeFixture<int8_t>,
    framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), ElementwiseQuantizedSignedDataset, ElementwiseQuantizationInfo, OutOfPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], abs_tolerance_qasymm8);
    }
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(
    ConfigureOnceUseFromDifferentThreads,
    CpuElementwiseMinQuantizedThreadSafeFixture<uint8_t>,
    framework::DatasetMode::ALL,
    combine(datasets::SmallShapes(), ElementwiseQuantizedDataset, ElementwiseQuantizationInfo, OutOfPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], abs_tolerance_qasymm8);
    }
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ThreadSafety
#endif           // #ifndef BARE_METAL
TEST_SUITE_END() // CpuElementwiseMin

TEST_SUITE(CpuPRelu)

template <typename T>
using CpuPReluFixture = CpuPReluValidationFixture<Tensor, Accessor, experimental::op::CpuPRelu, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuPReluFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ElementwiseFP32Dataset, InPlaceDataSet))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float
TEST_SUITE_END() // CpuPRelu

TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
