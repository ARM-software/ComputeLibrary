/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/MultiImage.h"
#include "arm_compute/runtime/NEON/functions/NEChannelExtract.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ChannelExtractFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// Input data sets
const auto ChannelExtractRGBADataset = combine(framework::dataset::make("FormatType", { Format::RGBA8888 }),
                                               framework::dataset::make("ChannelType", { Channel::R, Channel::G, Channel::B, Channel::A }));
const auto ChannelExtractYUVDataset = combine(framework::dataset::make("FormatType", { Format::YUYV422, Format::UYVY422 }),
                                              framework::dataset::make("ChannelType", { Channel::Y, Channel::U, Channel::V }));
const auto ChannelExtractYUVPlanarDataset = combine(framework::dataset::make("FormatType", { Format::IYUV, Format::YUV444, Format::NV12, Format::NV21 }),
                                                    framework::dataset::make("ChannelType", { Channel::Y, Channel::U, Channel::V }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ChannelExtract)

template <typename T>
using NEChannelExtractFixture = ChannelExtractValidationFixture<MultiImage, Tensor, Accessor, NEChannelExtract, T>;

TEST_SUITE(RGBA)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Small2DShapes(), ChannelExtractRGBADataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ChannelExtractRGBADataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // RGBA

TEST_SUITE(YUV)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Small2DShapes(), ChannelExtractYUVDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ChannelExtractYUVDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // YUV

TEST_SUITE(YUVPlanar)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Small2DShapes(), ChannelExtractYUVPlanarDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ChannelExtractYUVPlanarDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // YUVPlanar

TEST_SUITE_END() // ChannelExtract
TEST_SUITE_END() // Neon

} // namespace validation
} // namespace test
} // namespace arm_compute
