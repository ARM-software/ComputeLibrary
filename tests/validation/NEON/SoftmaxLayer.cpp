/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SoftmaxLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
constexpr AbsoluteTolerance<float> tolerance_f32(0.000001f);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<float> tolerance_f16(0.0001f);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC*/
/** Tolerance for fixed point operations */
constexpr AbsoluteTolerance<int16_t> tolerance_fixed_point(2);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
    DataType::QS8,
    DataType::QS16,
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(SoftmaxLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(concat(datasets::SoftmaxLayerSmallShapes(), datasets::SoftmaxLayerLargeShapes()), CNNDataTypes), shape, data_type)
{
    // Set fixed point position data type allowed
    const int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, data_type, 1, fixed_point_position);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NESoftmaxLayer smx_layer;
    smx_layer.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const int         step    = 16 / data_size_from_type(data_type);
    const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching data types
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching shapes
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 2), // Mismatching fixed point
                                            TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),
                                           }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 3),
                                            TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),
                                          })),
    framework::dataset::make("Expected", { false, false, false, true })),
    input_info, output_info, expected)
{
    bool is_valid = bool(NESoftmaxLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false)));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NESoftmaxLayerFixture = SoftmaxValidationFixture<Tensor, Accessor, NESoftmaxLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NESoftmaxLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                 framework::dataset::make("DataType", DataType::F16)),
                                                                                                         framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                               framework::dataset::make("DataType", DataType::F16)),
                                                                                                       framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NESoftmaxLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)),
                                                                                                          framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                        framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NESoftmaxLayerFixedPointFixture = SoftmaxValidationFixedPointFixture<Tensor, Accessor, NESoftmaxLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
// Testing for fixed point position [1,6) as reciprocal limits the maximum fixed point position to 5
FIXTURE_DATA_TEST_CASE(RunSmall, NESoftmaxLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SoftmaxLayerSmallShapes(), framework::dataset::make("DataType",
                                                                                                                     DataType::QS8)),
                                                                                                                     framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fixed_point);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                                   framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fixed_point);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14) as reciprocal limits the maximum fixed point position to 14
FIXTURE_DATA_TEST_CASE(RunSmall, NESoftmaxLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::QS16)),
                                                                                                                      framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fixed_point);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::QS16)),
                                                                                                                    framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fixed_point);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
