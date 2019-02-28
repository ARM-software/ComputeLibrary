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
#include "arm_compute/runtime/NEON/functions/NEChannelCombine.h"
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
#include "tests/validation/fixtures/ChannelCombineFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
inline void validate_configuration(const TensorShape &shape, Format format)
{
    const int num_planes = num_planes_from_format(format);

    // Create tensors
    MultiImage          dst     = create_multi_image<MultiImage>(shape, format);
    std::vector<Tensor> ref_src = create_tensor_planes<Tensor>(shape, format);

    // Create and configure function
    NEChannelCombine channel_combine;

    if(num_planes == 1)
    {
        const Tensor *tensor_extra = Format::RGBA8888 == format ? &ref_src[3] : nullptr;

        channel_combine.configure(&ref_src[0], &ref_src[1], &ref_src[2], tensor_extra, dst.plane(0));
    }
    else
    {
        channel_combine.configure(&ref_src[0], &ref_src[1], &ref_src[2], &dst);
    }

    // TODO(bsgcomp): Add validation for padding and shape (COMPMID-659)
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ChannelCombine)

TEST_SUITE(Configuration)
DATA_TEST_CASE(RGBA, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), framework::dataset::make("FormatType", { Format::RGB888, Format::RGBA8888 })),
               shape, format)
{
    validate_configuration(shape, format);
}
DATA_TEST_CASE(YUV, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), framework::dataset::make("FormatType", { Format::YUYV422, Format::UYVY422 })),
               shape, format)
{
    validate_configuration(shape, format);
}

DATA_TEST_CASE(YUVPlanar, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), framework::dataset::make("FormatType", { Format::IYUV, Format::YUV444, Format::NV12, Format::NV21 })),
               shape, format)
{
    validate_configuration(shape, format);
}
TEST_SUITE_END() // Configuration

template <typename T>
using NEChannelCombineFixture = ChannelCombineValidationFixture<MultiImage, Tensor, Accessor, NEChannelCombine, T>;

TEST_SUITE(RGBA)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelCombineFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("FormatType", { Format::RGB888, Format::RGBA8888 })))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelCombineFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("FormatType", { Format::RGB888, Format::RGBA8888 })))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // RGBA

TEST_SUITE(YUV)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelCombineFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("FormatType", { Format::YUYV422, Format::UYVY422 })))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelCombineFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("FormatType", { Format::YUYV422, Format::UYVY422 })))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // YUV

TEST_SUITE(YUVPlanar)
FIXTURE_DATA_TEST_CASE(RunSmall, NEChannelCombineFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("FormatType", { Format::NV12, Format::NV21, Format::IYUV, Format::YUV444 })))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEChannelCombineFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("FormatType", { Format::NV12, Format::NV21, Format::IYUV, Format::YUV444 })))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // YUVPlanar

TEST_SUITE_END() // ChannelCombine
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
