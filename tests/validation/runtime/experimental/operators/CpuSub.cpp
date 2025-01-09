/*
 * Copyright (c) 2017-2025 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuSub.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/cpu/kernels/CpuAddKernel.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/CpuArithmeticOperationsFixture.h"
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
const auto OutOfPlaceDataSet = make("InPlace", {false});

const auto ArithmeticSubtractionQuantizationInfoSignedDataset = combine(make("QuantizationInfoIn1", { QuantizationInfo(0.5f, 10) }),
                                                                                make("QuantizationInfoIn2", { QuantizationInfo(0.5f, 20) }),
                                                                        make("QuantizationInfoOut", { QuantizationInfo(0.5f, 50) }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuSub)

template <typename T>
using CpuSubFixture = CpuArithmeticSubtractionValidationFixture<Tensor, Accessor, experimental::op::CpuSub, T>;

template <typename T>
using CpuArithmeticSubtractionThreadSafeFixture = CpuArithmeticSubtractionThreadSafeValidationFixture<Tensor, Accessor,  experimental::op::CpuSub, T>;

template <typename T>
using CpuArithmeticSubtractionQuantizedThreadSafeFixture = CpuArithmeticSubtractionQuantizedThreadSafeValidationFixture<Tensor, Accessor,  experimental::op::CpuSub, T>;


TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(SmokeTest, CpuSubFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SmallShapes(),
        make("DataType", DataType::U8),
        make("ConvertPolicy", {ConvertPolicy::SATURATE, ConvertPolicy::WRAP}),
        OutOfPlaceDataSet
    ))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // U8

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuArithmeticSubtractionQuantizedThreadSafeFixture<int8_t>, framework::DatasetMode::ALL,
    combine(
        datasets::SmallShapes(),
        make("DataType", DataType::QASYMM8_SIGNED),
        make("ConvertPolicy", { ConvertPolicy::SATURATE }),
        ArithmeticSubtractionQuantizationInfoSignedDataset,
        OutOfPlaceDataSet
    ))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE(Integer)
TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuArithmeticSubtractionThreadSafeFixture<int32_t>, framework::DatasetMode::ALL,
    combine(
        datasets::TinyShapes(),
        make("DataType", DataType::S32),
        make("ConvertPolicy", {ConvertPolicy::WRAP}),
        OutOfPlaceDataSet
    ))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // S32
TEST_SUITE_END() // Integer

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuArithmeticSubtractionThreadSafeFixture<float>, framework::DatasetMode::ALL,
    combine(
        datasets::TinyShapes(),
        make("DataType", DataType::F32),
        make("ConvertPolicy", {ConvertPolicy::SATURATE}),
        OutOfPlaceDataSet
    ))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ThreadSafety
#endif // #ifndef BARE_METAL

TEST_SUITE_END() // CpuSub
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
