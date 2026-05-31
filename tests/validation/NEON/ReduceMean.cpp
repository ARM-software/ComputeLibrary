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
#include "arm_compute/runtime/NEON/functions/NEReduceMean.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SplitDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/ReduceMeanFixture.h"
#include "tests/validation/Validation.h"

#include <algorithm>
#include <cstdint>
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
constexpr AbsoluteTolerance<float> tolerance_f32(
    0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
#ifdef ARM_COMPUTE_ENABLE_FP16
constexpr AbsoluteTolerance<float> tolerance_f16(
    0.03f); /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
#endif // ARM_COMPUTE_ENABLE_FP16
#ifdef __aarch64__
constexpr AbsoluteTolerance<uint8_t> tolerance_u8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for unsigned 8-bit asymmetric quantized type */
constexpr AbsoluteTolerance<int8_t> tolerance_s8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for signed 8-bit asymmetric quantized type */
#else   // __aarch64__
constexpr AbsoluteTolerance<uint8_t> tolerance_u8(
    2); /**< Tolerance value for comparing reference's output against implementation's output for unsigned 8-bit asymmetric quantized type */
constexpr AbsoluteTolerance<int8_t> tolerance_s8(
    2); /**< Tolerance value for comparing reference's output against implementation's output for signed 8-bit asymmetric quantized type */
#endif  // __aarch64__

constexpr AbsoluteTolerance<uint8_t> zero_tolerance_u8(0);
constexpr AbsoluteTolerance<int8_t>  zero_tolerance_s8(0);

const auto axis_keep = combine(make("Axis",
                                    {Coordinates(0), Coordinates(1, 0), Coordinates(1, 2), Coordinates(0, 2),
                                     Coordinates(1, 3), Coordinates(2, 3), Coordinates(0, 1, 2, 3)}),
                               make("KeepDims", {true}));
const auto axis_drop =
    combine(make("Axis", {Coordinates(0), Coordinates(1), Coordinates(3)}), make("KeepDims", {false}));
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(ReduceMean)

TEST_CASE(ProperRoundingPolicyXReduction, framework::DatasetMode::ALL)
{
    // We do not need to stress vector and leftover loops diffrently
    // because the rounding is done scalarly at the end. Accumulation
    // is done over integer types.
    constexpr int x_len = 2;

    const auto input_shape  = TensorShape(x_len);
    const auto output_shape = TensorShape(1);
    const bool keep_dims    = true;
    const auto axis         = Coordinates(0);
    const auto input_qinfo  = QuantizationInfo(2 / 255.f, 0);
    const auto output_qinfo = QuantizationInfo(6 / 255.f, -1);
    const auto dtype        = DataType::QASYMM8_SIGNED;

    Tensor input  = create_tensor<Tensor>(input_shape, dtype, 1, input_qinfo);
    Tensor output = create_tensor<Tensor>(output_shape, dtype, 1, output_qinfo);

    NEReduceMean reduce_mean;
    reduce_mean.configure(&input, axis, keep_dims, &output);

    input.allocator()->allocate();
    output.allocator()->allocate();

    std::vector<int8_t> values{50, 26};
    library->fill_static_values(Accessor(input), values);

    std::vector<int8_t>  expected{12};
    SimpleTensor<int8_t> ref{output_shape, dtype, 1, input_qinfo};
    library->fill_static_values(ref, expected);

    reduce_mean.run();

    // The tolerance should be 0 because this test stresses the rounding behavior of the operator
    validate(Accessor(output), ref, zero_tolerance_s8);
}

#ifdef __aarch64__
// Due to the lack of instructions in a32, the rounding operation is less
// accurate
TEST_CASE(ProperRoundingPolicyNonXReduction, framework::DatasetMode::ALL)
{
    constexpr int x_len = 17; // > 16 to stress both vector and leftover loops

    const auto input_shape  = TensorShape(x_len, 2, 2, 1);
    const auto output_shape = TensorShape(x_len, 1, 1, 1);
    const bool keep_dims    = true;
    const auto axis         = Coordinates(1, 2);
    const auto input_qinfo  = QuantizationInfo(2 / 255.f, 127);
    const auto output_qinfo = QuantizationInfo(2 / 255.f, 127);
    const auto dtype        = DataType::QASYMM8;

    Tensor input  = create_tensor<Tensor>(input_shape, dtype, 1, input_qinfo);
    Tensor output = create_tensor<Tensor>(output_shape, dtype, 1, output_qinfo);

    NEReduceMean reduce_mean;
    reduce_mean.configure(&input, axis, keep_dims, &output);

    input.allocator()->allocate();
    output.allocator()->allocate();

    // {139, 139 ... 139 (x_len times) 154, 154, ... 154 (x_len_times) ...}
    std::vector<uint8_t> values;
    fill_n(back_inserter(values), x_len, 139);
    fill_n(back_inserter(values), x_len, 154);
    fill_n(back_inserter(values), x_len, 164);
    fill_n(back_inserter(values), x_len, 179);
    library->fill_static_values(Accessor(input), values);

    std::vector<uint8_t> expected;
    fill_n(back_inserter(expected), x_len, 159); // 159 = (139 + 154 + 164 + 179) / 4
    SimpleTensor<uint8_t> ref{output_shape, dtype, 1, input_qinfo};
    library->fill_static_values(ref, expected);

    reduce_mean.run();

    // The tolerance should be 0 because this test stresses the rounding behavior of the operator
    validate(Accessor(output), ref, zero_tolerance_u8);
}
#endif // __aarch64__

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
        make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid axis
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid output shape
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32),// OK
                                                TensorInfo(TensorShape{228U, 19U, 2U, 2U}, 1, DataType::F32),// OK
                                                TensorInfo(TensorShape{228U, 19U, 2U, 1U}, 1, DataType::F32) // Cannot support axis 3 not valid
        }),
        make("OutputInfo", { TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(32U, 16U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(19U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(19U), 1, DataType::F32)

        }),
        make("Axis", { Coordinates(4), Coordinates(0,2), Coordinates(2), Coordinates(3,2,0), Coordinates(3,2,0) }),
        make("Keep", { true, true, true, false, false }),
        make("Expected", { false, false, true, true, false })
        ),
        input_info, output_info, axis, keep, expected)
{
    const Status status = NEReduceMean::validate(&input_info.clone()->set_is_resizable(false), axis, keep, &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEReduceMeanFixture = ReduceMeanFixture<Tensor, Accessor, NEReduceMean, T>;

TEST_SUITE(Float)

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReduceMeanFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small4DShapes(),
                               make("DataType", DataType::F16),
                               concat(axis_keep, axis_drop)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReduceMeanFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large4DShapes(),
                               make("DataType", DataType::F16),
                               concat(axis_keep, axis_drop)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f16);
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
                       NEReduceMeanFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small4DShapes(),
                               make("DataType", DataType::F32),
                               concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReduceMeanFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large4DShapes(),
                               make("DataType", DataType::F32),
                               concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using NEReduceMeanQuantizedFixture = ReduceMeanQuantizedFixture<Tensor, Accessor, NEReduceMean, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReduceMeanQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small4DShapes(),
                               make("DataType", DataType::QASYMM8),
                               concat(axis_keep, axis_drop),
                               make("QuantizationInfoInput", {QuantizationInfo(1.f / 255, 5)}),
                               make("QuantizationInfoOutput", {QuantizationInfo(1.f / 255, 5)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}

TEST_SUITE(Requant)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReduceMeanQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small4DShapes(),
                               make("DataType", DataType::QASYMM8),
                               axis_drop,
                               make("QuantizationInfoInput", {QuantizationInfo(1.f / 255, 5)}),
                               make("QuantizationInfoOutput", {QuantizationInfo(1.f / 200, 16)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}
TEST_SUITE_END() // Requant

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReduceMeanQuantizedFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large4DShapes(),
                               make("DataType", DataType::QASYMM8),
                               concat(axis_keep, axis_drop),
                               make("QuantizationInfoInput", {QuantizationInfo(1.f / 255, 5)}),
                               make("QuantizationInfoOutput", {QuantizationInfo(1.f / 255, 5)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_u8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReduceMeanQuantizedFixture<int8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small4DShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               concat(axis_keep, axis_drop),
                               make("QuantizationInfoInput",
                                    {QuantizationInfo(1.f / 127, -10), QuantizationInfo(1.f / 250, -20)}),
                               make("QuantizationInfoInputOutput", {QuantizationInfo(1.f / 127, -10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}
TEST_SUITE(Requant)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReduceMeanQuantizedFixture<int8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small4DShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               axis_drop,
                               make("QuantizationInfoInput", {QuantizationInfo(1.f / 102, 2)}),
                               make("QuantizationInfoOutput", {QuantizationInfo(1.f / 113, 10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}
TEST_SUITE_END() // Requant

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReduceMeanQuantizedFixture<int8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large4DShapes(),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               concat(axis_keep, axis_drop),
                               make("QuantizationInfoInput", {QuantizationInfo(1.f / 127, -10)}),
                               make("QuantizationInfoInputOutput", {QuantizationInfo(1.f / 127, -10)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_s8);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ReduceMean
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
