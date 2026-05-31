/*
 * Copyright (c) 2019-2021, 2023-2026 Arm Limited.
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
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(0);
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_signed(0);
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(RoundLayer)

template <typename T>
using NERoundLayerFixture = RoundValidationFixture<Tensor, Accessor, NERoundLayer, T>;

template <typename T>
using NERoundLayerQuantizedFixture = RoundQuantizedValidationFixture<Tensor, Accessor, NERoundLayer, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERoundLayerFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NERoundLayerFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
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
                       NERoundLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NERoundLayerFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(), make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERoundLayerQuantizedFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8),
                               make("InputQInfo", {QuantizationInfo(0.2, -3)}),
                               make("OutputQInfo", {QuantizationInfo(0.5, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERoundLayerQuantizedFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("InputQInfo", {QuantizationInfo(0.075, 6)}),
                               make("OutputQInfo", {QuantizationInfo(0.1, -7)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // RoundLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
