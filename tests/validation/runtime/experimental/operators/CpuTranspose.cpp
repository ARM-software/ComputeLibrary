/*
 * Copyright (c) 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuTranspose.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuTransposeFixture.h"
#include "tests/validation/Validation.h"

/*
 * Tests for arm_compute::experimental::op::CpuTranspose which is a shallow wrapper for
 * arm_compute::cpu::CpuTranspose. Any future testing to the functionalities of cpu::CpuTranspose
 * will be tested in tests/NEON/Transpose.cpp given that op::CpuTranspose remain a shallow wrapper.
*/

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
using framework::dataset::make;

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)

TEST_SUITE(CpuTranspose)

template <typename T>
using CpuTransposeFixture = CpuTransposeValidationFixture<Tensor, Accessor, experimental::op::CpuTranspose, T>;

template <typename T>
using CpuTransposeThreadSafeFixture =
    CpuTransposeThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuTranspose, T>;

template <typename T>
using CpuTransposeQuantizedThreadSafeFixture =
    CpuTransposeQuantizedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuTranspose, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuTransposeFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(concat(datasets::Small1DShapes(), datasets::Small2DShapes()),
                               make("DataType", DataType::U8)))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() //U8

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuTransposeThreadSafeFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Small2DShapes(), make("DataType", DataType::F32)))
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
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuTransposeThreadSafeFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Tiny4DShapes(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i]);
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
                       CpuTransposeThreadSafeFixture<int32_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Tiny4DShapes(), make("DataType", DataType::S32)))
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
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuTransposeQuantizedThreadSafeFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Tiny4DShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("QuantizationInfoIn", {QuantizationInfo(0.5f, 0)})))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuTransposeQuantizedThreadSafeFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Tiny4DShapes(),
                               make("DataType", DataType::QASYMM8),
                               make("QuantizationInfoIn", {QuantizationInfo(0.5f, 0)})))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ThreadSafety
#endif           // #ifndef BARE_METAL
TEST_SUITE_END() // CpuTranspose

TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
