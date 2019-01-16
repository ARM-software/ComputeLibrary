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
#include "arm_compute/runtime/MultiImage.h"
#include "arm_compute/runtime/NEON/functions/NEColorConvert.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ColorConvertFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<uint8_t> tolerance_nv(2);

// Input data sets
const auto RGBDataset  = framework::dataset::make("FormatType", { Format::RGB888, Format::RGBA8888 });
const auto YUYVDataset = framework::dataset::make("FormatType", { Format::YUYV422, Format::UYVY422 });

const auto ColorConvert_RGBA_to_RGB = combine(framework::dataset::make("FormatType", { Format::RGBA8888 }),
                                              framework::dataset::make("FormatType", { Format::RGB888 }));

const auto ColorConvert_RGB_to_RGBA = combine(framework::dataset::make("FormatType", { Format::RGB888 }),
                                              framework::dataset::make("FormatType", { Format::RGBA8888 }));

const auto ColorConvert_RGB_to_U8 = combine(framework::dataset::make("FormatType", { Format::RGB888 }),
                                            framework::dataset::make("FormatType", { Format::U8 }));

const auto ColorConvert_YUYVDataset_to_RGBDataset = combine(YUYVDataset,
                                                            RGBDataset);

const auto ColorConvert_YUVPlanar_to_RGBDataset = combine(framework::dataset::make("FormatType", { Format::IYUV, Format::NV12, Format::NV21 }),
                                                          RGBDataset);

const auto ColorConvert_RGBDataset_to_NVDataset = combine(RGBDataset,
                                                          framework::dataset::make("FormatType", { Format::NV12, Format::IYUV, Format::YUV444 }));

const auto ColorConvert_YUYVDataset_to_NVDataset = combine(YUYVDataset,
                                                           framework::dataset::make("FormatType", { Format::NV12, Format::IYUV }));

const auto ColorConvert_NVDataset_to_YUVDataset = combine(framework::dataset::make("FormatType", { Format::NV12, Format::NV21 }),
                                                          framework::dataset::make("FormatType", { Format::IYUV, Format::YUV444 }));

inline void validate_configuration(const TensorShape &shape, Format src_format, Format dst_format)
{
    const unsigned int src_num_planes = num_planes_from_format(src_format);
    const unsigned int dst_num_planes = num_planes_from_format(dst_format);

    TensorShape input = adjust_odd_shape(shape, src_format);
    input             = adjust_odd_shape(input, dst_format);

    // Create tensors
    MultiImage ref_src = create_multi_image<MultiImage>(input, src_format);
    MultiImage ref_dst = create_multi_image<MultiImage>(input, dst_format);

    // Create and Configure function
    NEColorConvert color_convert;

    if(1U == src_num_planes)
    {
        const Tensor *src_plane = ref_src.plane(0);

        if(1U == dst_num_planes)
        {
            Tensor *dst_plane = ref_dst.plane(0);
            color_convert.configure(src_plane, dst_plane);
        }
        else
        {
            color_convert.configure(src_plane, &ref_dst);
        }
    }
    else
    {
        if(1U == dst_num_planes)
        {
            Tensor *dst_plane = ref_dst.plane(0);
            color_convert.configure(&ref_src, dst_plane);
        }
        else
        {
            color_convert.configure(&ref_src, &ref_dst);
        }
    }

    for(unsigned int plane_idx = 0; plane_idx < src_num_planes; ++plane_idx)
    {
        const Tensor *src_plane = ref_src.plane(plane_idx);

        ARM_COMPUTE_EXPECT(src_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
    }
    for(unsigned int plane_idx = 0; plane_idx < dst_num_planes; ++plane_idx)
    {
        const Tensor *dst_plane = ref_dst.plane(plane_idx);

        ARM_COMPUTE_EXPECT(dst_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
    }
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ColorConvert)

template <typename T>
using NEColorConvertFixture = ColorConvertValidationFixture<MultiImage, Tensor, Accessor, NEColorConvert, T>;

TEST_SUITE(Configuration)
DATA_TEST_CASE(RGBA, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_RGBA_to_RGB),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(RGB, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_RGB_to_RGBA),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(RGBtoU8, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_RGB_to_U8),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(YUV, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_YUYVDataset_to_RGBDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(YUVPlanar, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_YUVPlanar_to_RGBDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(NV, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_RGBDataset_to_NVDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(YUYVtoNV, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_YUYVDataset_to_NVDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(NVtoYUV, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(), ColorConvert_NVDataset_to_YUVDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}
TEST_SUITE_END() // Configuration

TEST_SUITE(RGBA)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGBA_to_RGB))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGBA_to_RGB))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // RGBA

TEST_SUITE(RGB)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGB_to_RGBA))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGB_to_RGBA))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // RGB

TEST_SUITE(RGBtoU8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGB_to_U8))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGB_to_U8))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // RGBtoU8

TEST_SUITE(YUV)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_YUYVDataset_to_RGBDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_YUYVDataset_to_RGBDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // YUV

TEST_SUITE(YUVPlanar)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_YUVPlanar_to_RGBDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_YUVPlanar_to_RGBDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // YUVPlanar

TEST_SUITE(NV)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGBDataset_to_NVDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx], tolerance_nv);
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGBDataset_to_NVDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx], tolerance_nv);
    }
}
TEST_SUITE_END() // NV

TEST_SUITE(YUYVtoNV)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_YUYVDataset_to_NVDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_YUYVDataset_to_NVDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // YUYVtoNV

TEST_SUITE(NVtoYUV)
FIXTURE_DATA_TEST_CASE(RunSmall, NEColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_NVDataset_to_YUVDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_NVDataset_to_YUVDataset))
{
    // Validate output
    for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
    {
        validate(Accessor(*_target.plane(plane_idx)), _reference[plane_idx]);
    }
}
TEST_SUITE_END() // NVtoYUV

TEST_SUITE_END() // ColorConvert
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
