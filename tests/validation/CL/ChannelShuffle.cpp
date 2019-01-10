/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLChannelShuffleLayer.h"
#include "tests/CL/CLAccessor.h"
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
TEST_SUITE(CL)
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
    ARM_COMPUTE_EXPECT(bool(CLChannelShuffleLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), num_groups)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallRandomChannelShuffleLayerDataset(), framework::dataset::make("DataType", { DataType::S8, DataType::U8, DataType::S16, DataType::U16, DataType::U32, DataType::S32, DataType::F16, DataType::F32 })),
               shape, num_groups, data_type)
{
    // Create tensors
    CLTensor ref_src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst     = create_tensor<CLTensor>(shape, data_type);

    // Create and Configure function
    CLChannelShuffleLayer channel_shuffle_func;
    channel_shuffle_func.configure(&ref_src, &dst, num_groups);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);
}

template <typename T>
using CLChannelShuffleLayerFixture = ChannelShuffleLayerValidationFixture<CLTensor, CLAccessor, CLChannelShuffleLayer, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLChannelShuffleLayerFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallRandomChannelShuffleLayerDataset(),
                                                                                                                   framework::dataset::make("DataType", DataType::U8)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLChannelShuffleLayerFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeRandomChannelShuffleLayerDataset(),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLChannelShuffleLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallRandomChannelShuffleLayerDataset(),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLChannelShuffleLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeRandomChannelShuffleLayerDataset(),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::F16)),
                                                                                                              framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLChannelShuffleLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallRandomChannelShuffleLayerDataset(),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLChannelShuffleLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeRandomChannelShuffleLayerDataset(),
                                                                                                                       framework::dataset::make("DataType",
                                                                                                                               DataType::F32)),
                                                                                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
