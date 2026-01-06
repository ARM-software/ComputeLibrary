/*
 * Copyright (c) 2018-2021, 2023-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/ElementwiseUnaryFixture.h"
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
RelativeTolerance<float> tolerance_fp32(0.000001f);
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<float> tolerance_fp16(0.01f);
#endif // ARM_COMPUTE_ENABLE_FP16
#if defined(__aarch64__)
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(0);
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_signed(0);
#else  // #if !defined(__aarch64__)
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(
    1); // There is difference of 1, because quantizing in reference uses round policy "TO_NEAREST_UP", where the armv7a neon kernel uses "TO_ZERO"
constexpr AbsoluteTolerance<int8_t> tolerance_qasymm8_signed(1);
#endif // #if !defined(__aarch64__)
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(RsqrtLayer)

template <typename T>
using CpuRsqrtDynamicShapeFloatFixture = RsqrtDynamicShapeFloatValidationFixture<Tensor, Accessor, NERsqrtLayer, T>;

template <typename T>
using CpuRsqrtDynamicShapeQuantizedFixture =
    RsqrtDynamicShapeQuantizedValidationFixture<Tensor, Accessor, NERsqrtLayer, T>;

/// Test test cases will execute the function with dynamic-stated shapes
/// Since other elementwise unary operations share the same kernel, this tests are added only here.
///
/// @note Only FP32 is tested for float since data type doesn't/shouldn't matter with dynamic shapes.
/// @note Only QASYMM8 is tested for quantized types since data type shouldn't matter with dynamic shapes.
/// Quantized types require separate testing because they sometimes use LUTs (look up table) under the hood.
/// If they hadn't been using LUTs, testing only the FP32 data type was enough because the kernel choice
/// does not matter when testing the dynamic shapes. It's only necessary to cover the different scenarios in the
/// configuration and run paths.

TEST_SUITE(DynamicShape)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuRsqrtDynamicShapeFloatFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}

TEST_SUITE_END() // FP32

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuRsqrtDynamicShapeQuantizedFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8),
                               make("InputQInfo", {QuantizationInfo(20, 0)}),
                               make("OutputQInfo", {QuantizationInfo(0.5, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // DynamicShape

template <typename T>
using NERsqrtLayerFixture = RsqrtValidationFixture<Tensor, Accessor, NERsqrtLayer, T>;

template <typename T>
using NERsqrtLayerQuantizedFixture = RsqrtQuantizedValidationFixture<Tensor, Accessor, NERsqrtLayer, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERsqrtLayerFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_fp16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NERsqrtLayerFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_fp16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

TEST_SUITE_END() // FP16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERsqrtLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERsqrtLayerQuantizedFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8),
                               make("InputQInfo", {QuantizationInfo(20, 0)}),
                               make("OutputQInfo", {QuantizationInfo(0.5, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERsqrtLayerQuantizedFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("InputQInfo", {QuantizationInfo(25, -128)}),
                               make("OutputQInfo", {QuantizationInfo(0.1, -7)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // RsqrtLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
