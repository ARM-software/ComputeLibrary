/*
 * Copyright (c) 2017-2021, 2024-2025 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuMul.h"

#include "arm_compute/core/Rounding.h"

#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/CpuMulFixture.h"
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
const float scale_255   = 1.f / 255.f;
const float scale_other = 1.f / 32768.f;
const float scale_unity = 1.f;

constexpr AbsoluteTolerance<float> tolerance_qsymm16(1); /**< Tolerance value for comparing reference's output against implementation's output for 16-bit quantized symmetric data types */
constexpr RelativeTolerance<float> tolerance_integer(1); /**< Tolerance value for comparing reference's output against implementation's output for integer data types */


/** Tests for in-place computation
 * With current interface storing TensorInfo with quantization information
 * in the kernel, it is difficult to have different tensor metadata
 * (e.g., quantization information, data type, different shape for broadcasting)
 * when an input is used as the output of the computation.
 * So, the following dataset for in-place computation is used only when
 * the exact same input and output Tensor object makes sense
 * (i.e., all the tensor metadata is the same) whereas if output is
 * expected to have either different quantization information, data type
 * or different shape we are not testing in-place computation.
 */
const auto OutOfPlaceDataSet = make("InPlace", {false});

const auto PixelWiseMultiplicationPolicySTNUDataset =   combine(
                                                            make("ConvertPolicy", { ConvertPolicy::SATURATE }),
                                                            make("RoundingPolicy", { RoundingPolicy::TO_NEAREST_UP })
                                                        );

const auto PixelWiseMultiplicationPolicySTZDataset =    combine(
                                                            make("ConvertPolicy", { ConvertPolicy::SATURATE }),
                                                            make("RoundingPolicy", { RoundingPolicy::TO_ZERO })
                                                        );

const auto PixelWiseMultiplicationQASYMM8QuantDataset = combine(
                                                            make("Src0QInfo", { QuantizationInfo(5.f / 32768.f, 0) }),
                                                            make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0) }),
                                                            make("OutQInfo", { QuantizationInfo(1.f / 32768.f, 0) })
                                                        );
} // namespace

using CpuMulU8U8toS16Fixture =
    CpuMulValidationFixture<Tensor, Accessor, experimental::op::CpuMul, uint8_t, uint8_t, int16_t>;

using CpuMulQS8QS8QS8ThreadSafeFixture =
    CpuMulQuantizedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuMul, int8_t, int8_t, int8_t>;

using CpuMulQ16Q16S32ThreadSafeFixture =
    CpuMulQuantizedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuMul, int16_t, int16_t, int32_t>;

using CpuMulQ16Q16Q16ThreadSafeFixture =
    CpuMulQuantizedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuMul, int16_t, int16_t, int16_t>;

using CpuMulS32S32S32ThreadSafeFixture =
    CpuMulThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuMul, int32_t, int32_t, int32_t>;

using CpuMulF32F32F32ThreadSafeFixture =
    CpuMulThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuMul, float, float, float>;

const auto PixelWiseMultiplicationQSYMM16QuantDataset = combine(
                                                                make("Src0QInfo", { QuantizationInfo(1.f / 32768.f, 0) }),
                                                                make("Src1QInfo", { QuantizationInfo(2.f / 32768.f, 0) }),
                                                                make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) }));
TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuMul)

TEST_SUITE(U8U8toS16)
FIXTURE_DATA_TEST_CASE(SmokeTest0, CpuMulU8U8toS16Fixture, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SmallShapes(),
        make("DataTypeIn1", DataType::U8),
        make("DataTypeIn2", DataType::U8),
        make("DataTypeOut", DataType::S16),
        make("Scale", {scale_255}),
        datasets::ConvertPolicies(),
        make("RoundingPolicy", RoundingPolicy::TO_NEAREST_UP),
        OutOfPlaceDataSet
    ))
{
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        // Validate output
        validate_wrap(Accessor(_target[i]), _reference[i], AbsoluteTolerance<int16_t>(1), 0.f);
    }
}

FIXTURE_DATA_TEST_CASE(SmokeTest1, CpuMulU8U8toS16Fixture, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SmallShapes(),
        make("DataTypeIn1", DataType::U8),
        make("DataTypeIn2", DataType::U8),
        make("DataTypeOut", DataType::S16),
        make("Scale", {scale_other}),
        datasets::ConvertPolicies(),
        make("RoundingPolicy", RoundingPolicy::TO_ZERO),
        OutOfPlaceDataSet
    ))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // U8U8toS16

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(QS8QS8QS8, CpuMulQS8QS8QS8ThreadSafeFixture, framework::DatasetMode::ALL,
    combine(
        datasets::SmallShapes(),
        make("DataTypeIn1", DataType::QASYMM8_SIGNED),
        make("DataTypeIn2", DataType::QASYMM8_SIGNED),
        make("DataTypeOut", DataType::QASYMM8_SIGNED),
        make("Scale", { scale_255 }),
        PixelWiseMultiplicationPolicySTNUDataset,
        PixelWiseMultiplicationQASYMM8QuantDataset,
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

TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(Q16Q16S32, CpuMulQ16Q16S32ThreadSafeFixture, framework::DatasetMode::ALL,
    combine(
        datasets::SmallShapes(),
        make("DataTypeIn1", DataType::QSYMM16),
        make("DataTypeIn2", DataType::QSYMM16),
        make("DataTypeOut", DataType::S32),
        make("Scale", { scale_unity }),
        PixelWiseMultiplicationPolicySTZDataset,
        PixelWiseMultiplicationQSYMM16QuantDataset,
        OutOfPlaceDataSet
    ))

{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}

FIXTURE_DATA_TEST_CASE(Q16Q16Q16, CpuMulQ16Q16Q16ThreadSafeFixture, framework::DatasetMode::ALL,
    combine(
        datasets::SmallShapes(),
        make("DataTypeIn1", DataType::QSYMM16),
        make("DataTypeIn2", DataType::QSYMM16),
        make("DataTypeOut", DataType::QSYMM16),
        make("Scale", { scale_unity }),
        PixelWiseMultiplicationPolicySTZDataset,
        PixelWiseMultiplicationQSYMM16QuantDataset,
        OutOfPlaceDataSet
    ))

{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_qsymm16);
    }
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized
TEST_SUITE(INTEGER)
TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(S32S32S32, CpuMulS32S32S32ThreadSafeFixture, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SmallShapes(),
        make("DataTypeIn1", DataType::S32),
        make("DataTypeIn2", DataType::S32),
        make("DataTypeOut", DataType::S32),
        make("Scale", {scale_other}),
        datasets::ConvertPolicies(),
        make("RoundingPolicy", RoundingPolicy::TO_ZERO),
        OutOfPlaceDataSet
    ))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_integer);
    }
}
TEST_SUITE_END() // S32
TEST_SUITE_END() // INTEGER

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(F32F32F32, CpuMulF32F32F32ThreadSafeFixture, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SmallShapes(),
        make("DataTypeIn1", DataType::F32),
        make("DataTypeIn2", DataType::F32),
        make("DataTypeOut", DataType::F32),
        make("Scale", {scale_other}),
        datasets::ConvertPolicies(),
        make("RoundingPolicy", RoundingPolicy::TO_ZERO),
        OutOfPlaceDataSet
    ))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_integer);
    }
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ThreadSafety
#endif // #ifndef BARE_METAL
TEST_SUITE_END() // CpuMul
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
