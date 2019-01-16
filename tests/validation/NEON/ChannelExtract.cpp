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

inline void validate_configuration(const TensorShape &shape, Format format, Channel channel)
{
    const unsigned int num_planes = num_planes_from_format(format);

    TensorShape dst_shape = adjust_odd_shape(shape, format);
    dst_shape             = calculate_subsampled_shape(dst_shape, format, channel);

    // Create tensors
    MultiImage ref_src = create_multi_image<MultiImage>(shape, format);
    Tensor     dst     = create_tensor<Tensor>(dst_shape, Format::U8);

    // Create and Configure function
    NEChannelExtract channel_extract;

    if(1U == num_planes)
    {
        const Tensor *plane_src = ref_src.plane(0);

        channel_extract.configure(plane_src, channel, &dst);
    }
    else
    {
        channel_extract.configure(&ref_src, channel, &dst);
    }

    // TODO(bsgcomp): Add validation for padding and shape (COMPMID-659)
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ChannelExtract)

template <typename T>
using NEChannelExtractFixture = ChannelExtractValidationFixture<MultiImage, Tensor, Accessor, NEChannelExtract, T>;

TEST_SUITE(Configuration)
DATA_TEST_CASE(RGBA, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ChannelExtractRGBADataset),
               shape, format, channel)
{
    validate_configuration(shape, format, channel);
}
DATA_TEST_CASE(YUV, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ChannelExtractYUVDataset),
               shape, format, channel)
{
    validate_configuration(shape, format, channel);
}

DATA_TEST_CASE(YUVPlanar, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ChannelExtractYUVPlanarDataset),
               shape, format, channel)
{
    validate_configuration(shape, format, channel);
}
TEST_SUITE_END() // Configuration

TEST_SUITE(RGBA)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ChannelExtractRGBADataset))
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ChannelExtractYUVDataset))
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelExtractFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ChannelExtractYUVPlanarDataset))
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
TEST_SUITE_END() // NEON

} // namespace validation
} // namespace test
} // namespace arm_compute
