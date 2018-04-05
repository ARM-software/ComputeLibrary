/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/CL/CLMultiImage.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLColorConvert.h"
#include "tests/CL/CLAccessor.h"
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
// Input data sets
const auto ColorConvertRGBADataset = combine(framework::dataset::make("FormatType", { Format::RGBA8888 }),
                                             framework::dataset::make("FormatType", { Format::RGB888 }));
const auto ColorConvertYUVDataset = combine(framework::dataset::make("FormatType", { Format::YUYV422, Format::UYVY422 }),
                                            framework::dataset::make("FormatType", { Format::RGB888, Format::RGBA8888 }));

const auto ColorConvertYUVPlanarDataset = combine(framework::dataset::make("FormatType", { Format::IYUV, Format::NV12, Format::NV21 }),
                                                  framework::dataset::make("FormatType", { Format::RGB888, Format::RGBA8888 }));

const auto ColorConvertRGBDataset = combine(framework::dataset::make("FormatType", { Format::RGB888 }),
                                            framework::dataset::make("FormatType", { Format::RGBA8888 }));

inline void validate_configuration(const TensorShape &shape, Format src_format, Format dst_format)
{
    const unsigned int src_num_planes = num_planes_from_format(src_format);

    const TensorShape dst_shape = adjust_odd_shape(shape, src_format);

    // Create tensors
    CLMultiImage ref_src = create_multi_image<CLMultiImage>(shape, src_format);
    CLTensor     dst     = create_tensor<CLTensor>(dst_shape, dst_format);

    // Create and Configure function
    CLColorConvert color_convert;

    if(1U == src_num_planes)
    {
        const CLTensor *plane_src = ref_src.cl_plane(0);

        color_convert.configure(plane_src, &dst);
        const ValidRegion dst_valid_region = shape_to_valid_region(dst_shape);
        validate(dst.info()->valid_region(), dst_valid_region);
    }
    else
    {
        color_convert.configure(&ref_src, &dst);
    }
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ColorConvert)

TEST_SUITE(Configuration)
DATA_TEST_CASE(RGBA, framework::DatasetMode::ALL, combine(concat(datasets::Small2DShapes(), datasets::Large2DShapes()), ColorConvertRGBADataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(YUV, framework::DatasetMode::ALL, combine(concat(datasets::Small2DShapes(), datasets::Large2DShapes()), ColorConvertYUVDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(YUVPlanar, framework::DatasetMode::ALL, combine(concat(datasets::Small2DShapes(), datasets::Large2DShapes()), ColorConvertYUVPlanarDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}

DATA_TEST_CASE(RGB, framework::DatasetMode::ALL, combine(concat(datasets::Small2DShapes(), datasets::Large2DShapes()), ColorConvertRGBDataset),
               shape, src_format, dst_format)
{
    validate_configuration(shape, src_format, dst_format);
}
TEST_SUITE_END()

template <typename T>
using CLColorConvertFixture = ColorConvertValidationFixture<CLMultiImage, CLTensor, CLAccessor, CLColorConvert, T>;

TEST_SUITE(RGBA)
FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvertRGBADataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvertRGBADataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(YUV)
FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvertYUVDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvertYUVDataset))
{
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(YUVPlanar)
FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvertYUVPlanarDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvertYUVPlanarDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(RGB)
FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvertRGBDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvertRGBDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
