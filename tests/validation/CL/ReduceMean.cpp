/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLReduceMean.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SplitDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReduceMeanFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float>   tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float>   tolerance_f16(0.03f);  /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);  /**< Tolerance value for comparing reference's output against implementation's output for 8-bit asymmetric quantized type */

const auto axis_keep = combine(framework::dataset::make("Axis", { Coordinates(0), Coordinates(1, 0), Coordinates(1, 2), Coordinates(0, 2), Coordinates(1, 3), Coordinates(0, 1, 2, 3) }),
                               framework::dataset::make("KeepDims", { true }));
const auto axis_drop = combine(framework::dataset::make("Axis", { Coordinates(0), Coordinates(1), Coordinates(3), Coordinates(1, 2), Coordinates(2, 1) }), framework::dataset::make("KeepDims", { false }));
} // namespace
TEST_SUITE(CL)
TEST_SUITE(ReduceMean)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid axis
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid output shape
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32),// OK
                                                TensorInfo(TensorShape{228U, 19U, 2U, 2U}, 1, DataType::F32),// OK
                                                TensorInfo(TensorShape{228U, 19U, 2U, 1U}, 1, DataType::F32) // Cannot support axis 3 not valid
        }),
        framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(32U, 16U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(19U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(19U), 1, DataType::F32)

        })),
        framework::dataset::make("Axis", { Coordinates(4), Coordinates(0,2), Coordinates(2), Coordinates(3,2,0), Coordinates(3,2,0) })),
        framework::dataset::make("Keep", { true, true, true, false, false })),
        framework::dataset::make("Expected", { false, false, true, true, false })),
        input_info, output_info, axis, keep, expected)
{
    const Status status = CLReduceMean::validate(&input_info.clone()->set_is_resizable(false), axis, keep, &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLReduceMeanFixture = ReduceMeanFixture<CLTensor, CLAccessor, CLReduceMean, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLReduceMeanFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F16)), concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLReduceMeanFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::F16)), concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLReduceMeanFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F32)), concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLReduceMeanFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::F32)), concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLReduceMeanQuantizedFixture = ReduceMeanQuantizedFixture<CLTensor, CLAccessor, CLReduceMean, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLReduceMeanQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), concat(axis_keep, axis_drop)), framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLReduceMeanQuantizedFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), concat(axis_keep, axis_drop)), framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLReduceMeanQuantizedFixture<int8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)), concat(axis_keep, axis_drop)), framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 102, 2) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLReduceMeanQuantizedFixture<int8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)), concat(axis_keep, axis_drop)), framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 102, 2) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ReduceMean
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
