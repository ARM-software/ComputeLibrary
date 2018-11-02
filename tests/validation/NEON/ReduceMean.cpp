/*
 * Copyright (c) 2018 ARM Limited.
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

#include "tests/NEON/Accessor.h"
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
constexpr AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<float> tolerance_f16(0.03f);   /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
#endif                                                     // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1); /**< Tolerance value for comparing reference's output against implementation's output for 8-bit asymmetric quantized type */

const auto axis_keep = combine(framework::dataset::make("Axis", { Coordinates(0), Coordinates(1, 0), Coordinates(1, 2), Coordinates(0, 2), Coordinates(1, 3), Coordinates(0, 1, 2, 3) }),
                               framework::dataset::make("KeepDims", { true }));
const auto axis_drop = combine(framework::dataset::make("Axis", { Coordinates(0), Coordinates(1), Coordinates(3) }), framework::dataset::make("KeepDims", { false }));
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(ReduceMean)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid axis
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid output shape
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32)
        }),
        framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(32U, 16U, 1U, 2U), 1, DataType::F32)
        })),
        framework::dataset::make("Axis", { Coordinates(4), Coordinates(0,2), Coordinates(2) })),
        framework::dataset::make("Expected", { false, false, true })),
        input_info, output_info, axis, expected)
{
    const Status status = NEReduceMean::validate(&input_info.clone()->set_is_resizable(false), axis, true, &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration,
               framework::DatasetMode::ALL,
               combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::F32 })),
               shape, data_type)
{
    // Create tensors
    Tensor ref_src = create_tensor<Tensor>(shape, data_type);
    Tensor dst;

    Coordinates axis(1);

    // Create and Configure function
    NEReduceMean reduce_mean;
    reduce_mean.configure(&ref_src, axis, true, &dst);

    // Validate valid region
    TensorShape output_shape = shape;
    output_shape.set(1, 1);
    const ValidRegion valid_region = shape_to_valid_region(output_shape);
    validate(dst.info()->valid_region(), valid_region);
}

template <typename T>
using NEReduceMeanFixture = ReduceMeanFixture<Tensor, Accessor, NEReduceMean, T>;

TEST_SUITE(Float)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReduceMeanFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F16)), concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReduceMeanFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::F16)), concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReduceMeanFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::F32)), concat(axis_keep, axis_drop)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReduceMeanFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::F32)), concat(axis_keep, axis_drop)))
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
                       combine(combine(combine(datasets::Small4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), concat(axis_keep, axis_drop)), framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255, 0) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReduceMeanQuantizedFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::Large4DShapes(), framework::dataset::make("DataType", DataType::QASYMM8)), concat(axis_keep, axis_drop)), framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255, 0) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ReduceMean
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
