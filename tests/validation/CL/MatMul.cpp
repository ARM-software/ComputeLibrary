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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLMatMul.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/DatasetModes.h"
#include "tests/framework/Macros.h"
#include "tests/framework/TestCase.h"
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
namespace
{
RelativeTolerance<float>             tolerance_f32(0.001f);      /**< Tolerance value for comparing reference's output against implementation's output for fp32 data type */
constexpr float                      abs_tolerance_f32(0.0001f); /**< Absolute tolerance value for comparing reference's output against implementation's output for fp32 data type in case using relative tolerance fails because of small values */
constexpr float                      abs_tolerance_f16(0.001f);  /**< Absolute tolerance value for comparing reference's output against implementation's output for fp16  data type in case using relative tolerance fails because of small values */
RelativeTolerance<half_float::half>  tolerance_f16(half(0.01));  /**< Tolerance value for comparing reference's output against implementation's output for fp16 data type */
constexpr AbsoluteTolerance<uint8_t> tolerance_quant(1);         /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
} // namespace

template <typename T>
using CLMatMulFixture = MatMulValidationFixture<CLTensor, CLAccessor, CLMatMul, GpuMatMulSettings, T>;

template <typename T>
using CLQuantizedMatMulFixture = QuantizedMatMulValidationFixture<CLTensor, CLAccessor, CLMatMul, GpuMatMulSettings, T>;

TEST_SUITE(CL)
TEST_SUITE(MatMul)

TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLMatMulFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallMatMulDataset(),
                                                                                                                    framework::dataset::make("TransposeA", { false, true })),
                                                                                                            framework::dataset::make("TransposeB", { false, true })),
                                                                                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLMatMulFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeMatMulDataset(),
                                                                                                                  framework::dataset::make("TransposeA", { false, true })),
                                                                                                          framework::dataset::make("TransposeB", { false, true })),
                                                                                                  framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLMatMulFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallMatMulDataset(),
                                                                                                                   framework::dataset::make("TransposeA", { false, true })),
                                                                                                           framework::dataset::make("TransposeB", { false, true })),
                                                                                                   framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLMatMulFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeMatMulDataset(),
                                                                                                                 framework::dataset::make("TransposeA", { false, true })),
                                                                                                         framework::dataset::make("TransposeB", { false, true })),
                                                                                                 framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}

TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)

FIXTURE_DATA_TEST_CASE(RunSmall, CLQuantizedMatMulFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(combine(
    datasets::SmallMatMulDataset(),
    framework::dataset::make("TransposeA", { false, true })),
    framework::dataset::make("TransposeB", { false, true })),
    framework::dataset::make("DataType", DataType::QASYMM8)),
    framework::dataset::make("ActivationInfo", { ActivationLayerInfo() })),
    framework::dataset::make("NumberOfExtraRuns", { 0, 1 })),
    framework::dataset::make("LhsQInfo", { QuantizationInfo(1.f / 50, 1) })),
    framework::dataset::make("RhsQInfo", { QuantizationInfo(1.f / 30, -1) })),
    framework::dataset::make("DstQInfo", { QuantizationInfo(1.f, 2) }))
)
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLQuantizedMatMulFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(combine(
    datasets::LargeMatMulDataset(),
    framework::dataset::make("TransposeA", { false, true })),
    framework::dataset::make("TransposeB", { false, true })),
    framework::dataset::make("DataType", DataType::QASYMM8)),
    framework::dataset::make("ActivationInfo", { ActivationLayerInfo() })),
    framework::dataset::make("NumberOfExtraRuns", { 0, 1 })),
    framework::dataset::make("LhsQInfo", { QuantizationInfo(1.f / 100, 1) })),
    framework::dataset::make("RhsQInfo", { QuantizationInfo(1.f / 200, -1) })),
    framework::dataset::make("DstQInfo", { QuantizationInfo(1.f, 2) }))
)
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)

FIXTURE_DATA_TEST_CASE(RunSmall, CLQuantizedMatMulFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(combine(
    datasets::SmallMatMulDataset(),
    framework::dataset::make("TransposeA", { false, true })),
    framework::dataset::make("TransposeB", { false, true })),
    framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
    framework::dataset::make("ActivationInfo", { ActivationLayerInfo() })),
    framework::dataset::make("NumberOfExtraRuns", { 0, 1 })),
    framework::dataset::make("LhsQInfo", { QuantizationInfo(1.f / 50, 1) })),
    framework::dataset::make("RhsQInfo", { QuantizationInfo(1.f / 30, -1) })),
    framework::dataset::make("DstQInfo", { QuantizationInfo(1.f, 2) }))
)
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLQuantizedMatMulFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(combine(
    datasets::LargeMatMulDataset(),
    framework::dataset::make("TransposeA", { false, true })),
    framework::dataset::make("TransposeB", { false, true })),
    framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
    framework::dataset::make("ActivationInfo", { ActivationLayerInfo() })),
    framework::dataset::make("NumberOfExtraRuns", { 0, 1 })),
    framework::dataset::make("LhsQInfo", { QuantizationInfo(1.f / 100, 1) })),
    framework::dataset::make("RhsQInfo", { QuantizationInfo(1.f / 200, -1) })),
    framework::dataset::make("DstQInfo", { QuantizationInfo(1.f, 2) }))
)
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_quant);
}

TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // Quantized

TEST_SUITE_END() // MatMul
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
