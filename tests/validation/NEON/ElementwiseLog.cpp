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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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

namespace
{
// Portable bit casting (pre-C++20).
uint32_t f32_bits(float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
float bits_f32(uint32_t u)
{
    float f = 0.f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Run the Log layer over explicit FP32 input bit patterns. NELogLayer dispatches to the SVE
// kernel on SVE-capable hardware and to the NEON kernel otherwise, so this exercises both paths.
std::vector<float> run_log_f32(const std::vector<uint32_t> &input_bits)
{
    const TensorShape shape(static_cast<unsigned int>(input_bits.size()));
    Tensor            src = create_tensor<Tensor>(shape, DataType::F32);
    Tensor            dst = create_tensor<Tensor>(shape, DataType::F32);

    NELogLayer layer;
    layer.configure(&src, &dst);
    src.allocator()->allocate();
    dst.allocator()->allocate();

    {
        Accessor acc(src);
        auto    *ptr = reinterpret_cast<uint32_t *>(acc.data());
        for (size_t i = 0; i < input_bits.size(); ++i)
        {
            ptr[i] = input_bits[i];
        }
    }

    layer.run();

    std::vector<float> out(input_bits.size());
    {
        Accessor    acc(dst);
        const auto *ptr = reinterpret_cast<const float *>(acc.data());
        for (size_t i = 0; i < out.size(); ++i)
        {
            out[i] = ptr[i];
        }
    }
    return out;
}

// Independent oracle: std::log follows IEEE-754 special-value rules on the host. Assert the
// output class explicitly rather than relying on the generic numeric comparator (which
// conflates NaN and infinity).
void validate_log_element(float got, uint32_t input_bits)
{
    const float ref = std::log(bits_f32(input_bits));
    if (std::isnan(ref))
    {
        ARM_COMPUTE_EXPECT(std::isnan(got), framework::LogLevel::ERRORS);
        const uint32_t bits = f32_bits(got);
        ARM_COMPUTE_EXPECT((bits & 0x7F800000u) == 0x7F800000u, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT((bits & 0x007FFFFFu) != 0u, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT((bits & 0x00400000u) != 0u, framework::LogLevel::ERRORS);
    }
    else if (std::isinf(ref))
    {
        ARM_COMPUTE_EXPECT(std::isinf(got), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(std::signbit(got) == std::signbit(ref), framework::LogLevel::ERRORS);
    }
    else
    {
        ARM_COMPUTE_EXPECT(std::isfinite(got), framework::LogLevel::ERRORS);
        // Finite accuracy of the unchanged polynomial is tightly covered by RunSmall/RunLarge;
        // here a generous-but-meaningful bound catches normalization/classification defects.
        const float tol = 1e-5f * std::max(1.0f, std::abs(ref));
        ARM_COMPUTE_EXPECT(std::abs(got - ref) <= tol, framework::LogLevel::ERRORS);
    }
}

void validate_log_vector(const std::vector<uint32_t> &input_bits)
{
    const std::vector<float> out = run_log_f32(input_bits);
    for (size_t i = 0; i < input_bits.size(); ++i)
    {
        validate_log_element(out[i], input_bits[i]);
    }
}

// Explicitly constructed FP32 bit patterns for IEEE exceptional and boundary inputs.
const std::vector<uint32_t> special_bits = {
    0x00000000u, // +0
    0x80000000u, // -0
    0x7F800000u, // +inf
    0xFF800000u, // -inf
    0x3F800000u, // +1.0
    0xBF800000u, // -1.0 (negative finite)
    0x40000000u, // +2.0
    0xC0000000u, // -2.0 (negative finite)
    0x7FC00000u, // qNaN
    0x7F800001u, // sNaN
    0x7FFFFFFFu, // NaN with maximal payload
    0x00000001u, // smallest positive subnormal
    0x00000002u, // second positive subnormal
    0x0000007Fu, // representative middle subnormal
    0x003FFFFFu, // large subnormal
    0x007FFFFFu, // largest positive subnormal
    0x00800000u, // smallest positive normal
    0x7F7FFFFFu, // largest positive normal
    0x3F000000u, // 0.5
    0x40490FDBu, // ~pi
};

constexpr uint32_t normal_fill_bits = 0x40000000u; // 2.0f
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(LogLayer)

template <typename T>
using NELogLayerFixture = LogValidationFixture<Tensor, Accessor, NELogLayer, T>;

template <typename T>
using NELogLayerQuantizedFixture = LogQuantizedValidationFixture<Tensor, Accessor, NELogLayer, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NELogLayerFixture<half>,
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
                       NELogLayerFixture<half>,
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
                       NELogLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NELogLayerFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(), make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE(IEEEExceptional)

// All exceptional/boundary patterns in one mixed vector, then each in isolation (length 1
// exercises the scalar remainder path).
TEST_CASE(SpecialValues, framework::DatasetMode::PRECOMMIT)
{
    validate_log_vector(special_bits);
    for (const uint32_t b : special_bits)
    {
        validate_log_vector({b});
    }
}

// Lengths around the 4-wide NEON and 16-wide SVE vector steps, including scalar remainders.
// A cycling pattern places exceptional, subnormal and normal values at varying lane alignments.
DATA_TEST_CASE(VectorBoundaries,
               framework::DatasetMode::PRECOMMIT,
               make("Length", {3U, 4U, 5U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U, 33U, 63U, 64U, 65U}),
               length)
{
    std::vector<uint32_t> input(length);
    for (unsigned int i = 0; i < length; ++i)
    {
        input[i] = special_bits[i % special_bits.size()];
    }
    validate_log_vector(input);
}

// Place every special value in every lane position of 4- and 8-element vectors (other lanes a
// positive normal), proving no lane-position-dependent leakage and no contamination of neighbours.
TEST_CASE(LanePositionSweep, framework::DatasetMode::PRECOMMIT)
{
    for (const unsigned int len : {4U, 8U, 16U, 32U, 64U})
    {
        for (const uint32_t s : special_bits)
        {
            for (unsigned int pos = 0; pos < len; ++pos)
            {
                std::vector<uint32_t> input(len, normal_fill_bits);
                input[pos] = s;
                validate_log_vector(input);
            }
        }
    }
}

// Hand-crafted vectors mixing normal, subnormal and exceptional lanes simultaneously.
TEST_CASE(MixedLanes, framework::DatasetMode::PRECOMMIT)
{
    validate_log_vector({0x00000001u, 0x3F800000u, 0xFF800000u, 0x00000000u}); // min-sub, 1.0, -inf, +0
    validate_log_vector({0x40000000u, 0x007FFFFFu, 0x7FC00000u, 0xC0000000u}); // 2.0, max-sub, qNaN, -2.0
    validate_log_vector({0x00800000u, 0x80000000u, 0x7F800000u, 0x003FFFFFu}); // min-normal, -0, +inf, sub
    validate_log_vector({0x7F800001u, 0x3F000000u, 0x00000002u, 0xBF800000u}); // sNaN, 0.5, sub, -1.0
    validate_log_vector({0x7F7FFFFFu, 0x0000007Fu, 0x40490FDBu, 0x7FFFFFFFu}); // max-normal, sub, ~pi, NaN
}

// Subnormal normalization must be accurate: large |log| makes the relative error tiny.
TEST_CASE(SubnormalAccuracy, framework::DatasetMode::PRECOMMIT)
{
    const std::vector<uint32_t> subnormals = {0x00000001u, 0x00000002u, 0x0000007Fu,
                                              0x00000100u, 0x003FFFFFu, 0x007FFFFFu};
    const std::vector<float>    out        = run_log_f32(subnormals);
    for (size_t i = 0; i < subnormals.size(); ++i)
    {
        const float ref = std::log(bits_f32(subnormals[i]));
        ARM_COMPUTE_EXPECT(std::isfinite(out[i]), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(std::abs(out[i] - ref) <= 1e-6f * std::abs(ref), framework::LogLevel::ERRORS);
    }
}

// Larger tensors with a repeating exceptional/subnormal/normal pattern.
DATA_TEST_CASE(LargeTensor, framework::DatasetMode::NIGHTLY, make("Length", {1024U, 4097U, 65537U}), length)
{
    std::vector<uint32_t> input(length);
    for (unsigned int i = 0; i < length; ++i)
    {
        input[i] = special_bits[i % special_bits.size()];
    }
    validate_log_vector(input);
}

TEST_SUITE_END() // IEEEExceptional

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NELogLayerQuantizedFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8),
                               make("InputQInfo", {QuantizationInfo(10.5, 0), QuantizationInfo(0.5, -10)}),
                               make("OutputQInfo", {QuantizationInfo(5, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NELogLayerQuantizedFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("InputQInfo", {QuantizationInfo(0.75, -128)}),
                               make("OutputQInfo", {QuantizationInfo(12.5, -2)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // LogLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
