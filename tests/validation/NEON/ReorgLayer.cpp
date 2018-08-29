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
#include "arm_compute/runtime/NEON/functions/NEReorgLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReorgLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(ReorgLayer)

DATA_TEST_CASE(Configuration,
               framework::DatasetMode::ALL,
               combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                       framework::dataset::make("Stride", { 2, 3 })),
                               framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F16, DataType::F32 })),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
               shape, stride, data_type, data_layout)
{
    // Create tensors
    Tensor ref_src = create_tensor<Tensor>(shape, data_type, 1, QuantizationInfo(), data_layout);
    Tensor dst;

    // Create and Configure function
    NEReorgLayer reorg_func;
    reorg_func.configure(&ref_src, &dst, stride);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(dst.info()->tensor_shape());
    validate(dst.info()->valid_region(), valid_region);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo",{
                TensorInfo(TensorShape(8U, 8U, 5U, 3U), 1, DataType::U16),     // Invalid stride
                TensorInfo(TensorShape(8U, 8U, 5U, 3U), 1, DataType::U16),     // Invalid output shape
                TensorInfo(TensorShape(8U, 8U, 5U, 3U), 1, DataType::U16),     // valid
        }),
        framework::dataset::make("OutputInfo", {
                TensorInfo(TensorShape(4U, 4U, 20U, 3U), 1, DataType::U16),
                TensorInfo(TensorShape(4U, 4U, 10U, 3U), 1, DataType::U16),
                TensorInfo(TensorShape(4U, 4U, 20U, 3U), 1, DataType::U16),
        })),
        framework::dataset::make("Stride", { -1, 2, 2 })),
        framework::dataset::make("Expected", { false, false, true })),
        input_info, output_info, stride, expected)
{
    Status status = NEReorgLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), stride);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEReorgLayerFixture = ReorgLayerValidationFixture<Tensor, Accessor, NEReorgLayer, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReorgLayerFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("Stride", { 2, 3 })),
                                       framework::dataset::make("DataType", DataType::U8)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReorgLayerFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("Stride", { 2, 3 })),
                                       framework::dataset::make("DataType", DataType::U8)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReorgLayerFixture<uint16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("Stride", { 2, 3 })),
                                       framework::dataset::make("DataType", DataType::U16)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReorgLayerFixture<uint16_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("Stride", { 2, 3 })),
                                       framework::dataset::make("DataType", DataType::U16)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReorgLayerFixture<uint32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("Stride", { 2, 3 })),
                                       framework::dataset::make("DataType", DataType::U32)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReorgLayerFixture<uint32_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("Stride", { 2, 3 })),
                                       framework::dataset::make("DataType", DataType::U32)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U32

TEST_SUITE_END() // ReorgLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
