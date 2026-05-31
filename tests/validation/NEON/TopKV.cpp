/*
 * Copyright (c) 2026 Arm Limited.
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
#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/runtime/CPP/functions/CPPTopKV.h"
#include "arm_compute/runtime/NEON/functions/NETopKV.h"
#include "arm_compute/runtime/Tensor.h"

#include "tests/datasets/TopKVDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/TopKVLayerFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

const auto tiny_dataset_topkv  = combine(datasets::Small1DTopKV(), make("K", 1));
const auto small_dataset_topkv = combine(datasets::SmallTopKV(), make("K", 3, 5));
const auto large_dataset_topkv = combine(datasets::LargeTopKV(), make("K", 3, 5));
const auto s32_small_dataset   = combine(small_dataset_topkv, make("DataType", DataType::S32));
const auto s32_tiny_dataset    = combine(tiny_dataset_topkv, make("DataType", DataType::S32));
const auto s32_large_dataset   = combine(large_dataset_topkv, make("DataType", DataType::S32));
const auto f32_small_dataset   = combine(small_dataset_topkv, make("DataType", DataType::F32));
const auto f32_tiny_dataset    = combine(tiny_dataset_topkv, make("DataType", DataType::F32));
const auto f32_large_dataset   = combine(large_dataset_topkv, make("DataType", DataType::F32));
const auto f16_small_dataset   = combine(small_dataset_topkv, make("DataType", DataType::F16));
const auto f16_tiny_dataset    = combine(tiny_dataset_topkv, make("DataType", DataType::F16));
const auto f16_large_dataset   = combine(large_dataset_topkv, make("DataType", DataType::F16));
const auto qu8_small_dataset   = combine(small_dataset_topkv, make("DataType", DataType::QASYMM8));
const auto qu8_tiny_dataset    = combine(tiny_dataset_topkv, make("DataType", DataType::QASYMM8));
const auto qu8_large_dataset   = combine(large_dataset_topkv, make("DataType", DataType::QASYMM8));
const auto qs8_small_dataset   = combine(small_dataset_topkv, make("DataType", DataType::QASYMM8_SIGNED));
const auto qs8_tiny_dataset    = combine(tiny_dataset_topkv, make("DataType", DataType::QASYMM8));
const auto qs8_large_dataset   = combine(large_dataset_topkv, make("DataType", DataType::QASYMM8_SIGNED));

constexpr AbsoluteTolerance<uint8_t> ZeroTolerance{0};

TEST_SUITE(NEON)
TEST_SUITE(TopKVLayer)

template <typename T>
using NETopKVFixture = TopKVValidationFixture<Tensor, Accessor, NETopKV, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunTiny, NETopKVFixture<uint8_t>, framework::DatasetMode::ALL, qu8_tiny_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NETopKVFixture<uint8_t>, framework::DatasetMode::ALL, qu8_small_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETopKVFixture<uint8_t>, framework::DatasetMode::NIGHTLY, qu8_large_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunTiny, NETopKVFixture<int8_t>, framework::DatasetMode::ALL, qs8_tiny_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NETopKVFixture<int8_t>, framework::DatasetMode::ALL, qs8_small_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETopKVFixture<int8_t>, framework::DatasetMode::NIGHTLY, qs8_large_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunTiny, NETopKVFixture<half>, framework::DatasetMode::ALL, f16_tiny_dataset)
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, ZeroTolerance);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunSmall, NETopKVFixture<half>, framework::DatasetMode::ALL, f16_small_dataset)
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, ZeroTolerance);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETopKVFixture<half>, framework::DatasetMode::NIGHTLY, f16_large_dataset)
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, ZeroTolerance);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunTiny, NETopKVFixture<float>, framework::DatasetMode::ALL, f32_tiny_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NETopKVFixture<float>, framework::DatasetMode::ALL, f32_small_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETopKVFixture<float>, framework::DatasetMode::NIGHTLY, f32_large_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunTiny, NETopKVFixture<int32_t>, framework::DatasetMode::ALL, s32_tiny_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NETopKVFixture<int32_t>, framework::DatasetMode::ALL, s32_small_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETopKVFixture<int32_t>, framework::DatasetMode::NIGHTLY, s32_large_dataset)
{
    // Validate output
    validate(Accessor(_target), _reference, ZeroTolerance);
}
TEST_SUITE_END() // S32

TEST_SUITE_END() // TopKVLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
