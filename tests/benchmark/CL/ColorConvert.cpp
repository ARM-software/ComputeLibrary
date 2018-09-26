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
#include "arm_compute/runtime/CL/CLMultiImage.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLColorConvert.h"
#include "tests/CL/CLAccessor.h"
#include "tests/benchmark/fixtures/ColorConvertFixture.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
namespace
{
const auto RGBDataset  = framework::dataset::make("FormatType", { Format::RGB888, Format::RGBA8888 });
const auto YUYVDataset = framework::dataset::make("FormatType", { Format::YUYV422, Format::UYVY422 });

const auto ColorConvert_RGBA_to_RGB = combine(framework::dataset::make("FormatType", { Format::RGBA8888 }),
                                              framework::dataset::make("FormatType", { Format::RGB888 }));

const auto ColorConvert_RGB_to_RGBA = combine(framework::dataset::make("FormatType", { Format::RGB888 }),
                                              framework::dataset::make("FormatType", { Format::RGBA8888 }));

const auto ColorConvert_RGB_to_U8 = combine(framework::dataset::make("FormatType", { Format::RGB888 }),
                                            framework::dataset::make("FormatType", { Format::U8 }));

const auto ColorConvert_YUYV_to_RGBDataset = combine(YUYVDataset,
                                                     RGBDataset);

const auto ColorConvert_YUVPlanar_to_RGBDataset = combine(framework::dataset::make("FormatType", { Format::IYUV, Format::NV12, Format::NV21 }),
                                                          RGBDataset);

const auto ColorConvert_RGBDataset_to_NVDataset = combine(RGBDataset,
                                                          framework::dataset::make("FormatType", { Format::NV12, Format::IYUV, Format::YUV444 }));

const auto ColorConvert_YUYVDataset_to_NVDataset = combine(YUYVDataset,
                                                           framework::dataset::make("FormatType", { Format::NV12, Format::IYUV }));

const auto ColorConvert_NVDataset_to_YUVDataset = combine(framework::dataset::make("FormatType", { Format::NV12, Format::NV21 }),
                                                          framework::dataset::make("FormatType", { Format::IYUV, Format::YUV444 }));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ColorConvert)

using CLColorConvertFixture = ColorConvertFixture<CLMultiImage, CLTensor, CLAccessor, CLColorConvert>;

TEST_SUITE(RGBA)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGBA_to_RGB));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGBA_to_RGB));
TEST_SUITE_END()

TEST_SUITE(RGB)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGB_to_RGBA));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGB_to_RGBA));
TEST_SUITE_END()

TEST_SUITE(RGBtoU8)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGB_to_U8));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGB_to_U8));
TEST_SUITE_END()

TEST_SUITE(YUV)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_YUYV_to_RGBDataset));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_YUYV_to_RGBDataset));
TEST_SUITE_END()

TEST_SUITE(YUVPlanar)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_YUVPlanar_to_RGBDataset));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_YUVPlanar_to_RGBDataset));
TEST_SUITE_END()

TEST_SUITE(NV)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_RGBDataset_to_NVDataset));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_RGBDataset_to_NVDataset));
TEST_SUITE_END()

TEST_SUITE(YUYVtoNV)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_YUYVDataset_to_NVDataset));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_YUYVDataset_to_NVDataset));
TEST_SUITE_END()

TEST_SUITE(NVtoYUV)
// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLColorConvertFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), ColorConvert_NVDataset_to_YUVDataset));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLColorConvertFixture, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), ColorConvert_NVDataset_to_YUVDataset));
TEST_SUITE_END()

TEST_SUITE_END() // ColorConvert
TEST_SUITE_END() // CL
} // namespace benchmark
} // namespace test
} // namespace arm_compute
