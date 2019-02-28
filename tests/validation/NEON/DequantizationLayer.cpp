/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DequantizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
constexpr AbsoluteTolerance<float> tolerance_f32(0.001f);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DequantizationLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),  // Wrong input data type
                                                       TensorInfo(TensorShape(16U, 5U, 16U), 1, DataType::U8),  // Invalid shape
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::U8),  // Wrong output data type
                                                       TensorInfo(TensorShape(16U, 16U, 2U, 5U), 1, DataType::U8),  // Missmatching shapes
                                                       TensorInfo(TensorShape(17U, 16U, 16U, 5U), 1, DataType::U8),  // Shrink window
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::U8),  // Valid
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 5U, 16U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(17U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("MinMax",{ TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(2U), 1, DataType::U8),
                                                     })),
               framework::dataset::make("Expected", { false, false, false, false, false, true})),
               input_info, output_info, min_max, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEDequantizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), &min_max.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::Small3DShapes(), framework::dataset::make("DataType", DataType::U8)), shape, data_type)
{
    TensorShape shape_min_max = shape;
    shape_min_max.set(Window::DimX, 2);

    // Remove Y and Z dimensions and keep the batches
    shape_min_max.remove_dimension(1);
    shape_min_max.remove_dimension(1);

    // Create tensors
    Tensor src     = create_tensor<Tensor>(shape, data_type);
    Tensor dst     = create_tensor<Tensor>(shape, DataType::F32);
    Tensor min_max = create_tensor<Tensor>(shape_min_max, DataType::F32);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(min_max.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEDequantizationLayer dequant_layer;
    dequant_layer.configure(&src, &dst, &min_max);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate valid region of min_max tensor
    const ValidRegion valid_region_min_max = shape_to_valid_region(shape_min_max);
    validate(min_max.info()->valid_region(), valid_region_min_max);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 8).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);

    // Validate padding of min_max tensor
    const PaddingSize padding_min_max = PaddingCalculator(shape_min_max.x(), 2).required_padding();
    validate(min_max.info()->padding(), padding_min_max);
}

template <typename T>
using NEDequantizationLayerFixture = DequantizationValidationFixture<Tensor, Accessor, NEDequantizationLayer, T>;

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDequantizationLayerFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                                                                                                                   framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDequantizationLayerFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                                                                                                 framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // U8
TEST_SUITE_END() // Integer

TEST_SUITE_END() // DequantizationLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
