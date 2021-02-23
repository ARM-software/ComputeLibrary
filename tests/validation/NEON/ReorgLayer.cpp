/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "tests/datasets/ReorgLayerDataset.h"
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

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::S64),    // Wrong output tensor
                                                       TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32),    // Wrong output tensor
                                                       TensorInfo(TensorShape(3U, 12U, 4U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(3U, 12U, 4U, 2U), 1, DataType::F32),     // Wrong data type
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::S64),
                                                       TensorInfo(TensorShape(5U, 6U, 4U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 6U, 2, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(1U, 4U, 36U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(1U, 4U, 36U, 2U), 1, DataType::F16),
                                                     })),
               framework::dataset::make("Stride", { 2, 2, 4, 3 })),
               framework::dataset::make("Expected", { false, true, false, true, false })),
               input_info, output_info, stride, expected)
{
    bool status = bool(NEReorgLayer::validate(&input_info, &output_info, stride));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEReorgLayerFixture = ReorgLayerValidationFixture<Tensor, Accessor, NEReorgLayer, T>;

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReorgLayerFixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                  DataType::S32)),
                                                                                                          framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEReorgLayerFixture<int32_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                DataType::S32)),
                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReorgLayerFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                  DataType::S16)),
                                                                                                          framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEReorgLayerFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                DataType::S16)),
                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReorgLayerFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                 DataType::S8)),
                                                                                                         framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEReorgLayerFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeReorgLayerDataset(), framework::dataset::make("DataType", DataType::S8)),
                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S8

TEST_SUITE_END() // ReorgLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
