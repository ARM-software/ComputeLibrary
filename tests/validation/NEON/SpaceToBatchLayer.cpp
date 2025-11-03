/*
 * Copyright (c) 2019-2021, 2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NESpaceToBatchLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SpaceToBatchDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SpaceToBatchFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

TEST_SUITE(NEON)
TEST_SUITE(SpaceToBatchLayer)

template <typename T>
using NESpaceToBatchLayerFixture = SpaceToBatchLayerValidationFixture<Tensor, Accessor, NESpaceToBatchLayer, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("InputInfo", { TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),    // Wrong data type block shape
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U, 4U), 1, DataType::F32),    // Wrong tensor shape
                                                     }),
               make("BlockShapeInfo",{ TensorInfo(TensorShape(2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(2U), 1, DataType::S32),
                                                     }),
               make("PaddingsShapeInfo",{ TensorInfo(TensorShape(2U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(2U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(2U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(2U, 2U), 1, DataType::S32),
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),
                                                     }),
               make("Expected", { true, false, false, false})
               ),
               input_info, block_shape_info, paddings_info, output_info, expected)
{
    bool has_error = bool(NESpaceToBatchLayer::validate(&input_info.clone()->set_is_resizable(false), &block_shape_info.clone()->set_is_resizable(false), &paddings_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false)));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
DATA_TEST_CASE(ValidateStatic, framework::DatasetMode::ALL, zip(
               make("InputInfo", { TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F32),    // Negative block shapes
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U, 4U), 1, DataType::F32), // Wrong tensor shape
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U, 4U), 1, DataType::F32), // Wrong paddings
                                                     }),
               make("BlockShapeX", { 2, 2, 2, 2, 2 }),
               make("BlockShapeY", { 2, 2, -2, 2, 2 }),
               make("PadLeft", { Size2D(0, 0), Size2D(0, 0), Size2D(0, 0), Size2D(0, 0), Size2D(3, 11) }),
               make("PadRight", { Size2D(0, 0), Size2D(0, 0), Size2D(0, 0), Size2D(0, 0), Size2D(3, 11) }),
               make("OutputInfo",{ TensorInfo(TensorShape(16U, 8U, 2U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 8U, 2U, 4U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 8U, 2U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 8U, 2U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 8U, 2U, 4U), 1, DataType::F32),
                                                     }),
               make("Expected", { true, false, false, false, false})
               ),
               input_info, block_shape_x, block_shape_y, padding_left, padding_right, output_info, expected)
{
    bool has_error = bool(NESpaceToBatchLayer::validate(&input_info.clone()->set_is_resizable(false), block_shape_x, block_shape_y, padding_left, padding_right, &output_info.clone()->set_is_resizable(false)));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(Small, NESpaceToBatchLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallSpaceToBatchLayerDataset(), make("DataType",
                                                                                                                    DataType::F32),
                                                                                                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(Large, NESpaceToBatchLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeSpaceToBatchLayerDataset(), make("DataType",
                                                                                                                  DataType::F32),
                                                                                                          make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Small, NESpaceToBatchLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallSpaceToBatchLayerDataset(),
                                                                                                                   make("DataType", DataType::F16),
                                                                                                           make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(Large, NESpaceToBatchLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeSpaceToBatchLayerDataset(),
                                                                                                                 make("DataType", DataType::F16),
                                                                                                         make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

template <typename T>
using NESpaceToBatchLayerQuantizedFixture = SpaceToBatchLayerValidationQuantizedFixture<Tensor, Accessor, NESpaceToBatchLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(Small, NESpaceToBatchLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallSpaceToBatchLayerDataset(),
                                                                                                                       make("DataType", DataType::QASYMM8),
                                                                                                                       make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }),
                                                                                                                       make("QuantizationInfo", { 1.f / 255.f, 9.f })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(Large, NESpaceToBatchLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeSpaceToBatchLayerDataset(),
                                                                                                                     make("DataType", DataType::QASYMM8),
                                                                                                                     make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }),
                                                                                                                     make("QuantizationInfo", { 1.f / 255.f, 9.f })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // SpaceToBatch
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
