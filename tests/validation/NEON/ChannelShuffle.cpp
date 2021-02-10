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
#include "arm_compute/runtime/NEON/functions/NEChannelShuffleLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ChannelShuffleLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ChannelShuffleLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(ChannelShuffle)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),  // Invalid num groups
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::U8),  // Mismatching data_type
                                                       TensorInfo(TensorShape(4U, 5U, 4U), 1, DataType::F32),  // Mismatching shapes
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),  // Num groups == channels
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),  // (channels % num_groups) != 0
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),  // Valid
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 4U, 4U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("NumGroups",{ 1, 2, 2, 4, 3, 2,
                                                     })),
               framework::dataset::make("Expected", { false, false, false, false, false, true})),
               input_info, output_info, num_groups, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEChannelShuffleLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), num_groups)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEChannelShuffleLayerFixture = ChannelShuffleLayerValidationFixture<Tensor, Accessor, NEChannelShuffleLayer, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelShuffleLayerFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallRandomChannelShuffleLayerDataset(),
                                                                                                                   framework::dataset::make("DataType", DataType::U8)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelShuffleLayerFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeRandomChannelShuffleLayerDataset(),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelShuffleLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallRandomChannelShuffleLayerDataset(),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelShuffleLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeRandomChannelShuffleLayerDataset(),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::F16)),
                                                                                                              framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelShuffleLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallRandomChannelShuffleLayerDataset(),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelShuffleLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeRandomChannelShuffleLayerDataset(),
                                                                                                                       framework::dataset::make("DataType",
                                                                                                                               DataType::F32)),
                                                                                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ChannelShuffle
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
