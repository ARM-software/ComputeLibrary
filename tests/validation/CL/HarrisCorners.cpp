/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLHarrisCorners.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/CLArrayAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ImageFileDatasets.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/HarrisCornersFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto data_nightly   = combine(framework::dataset::make("GradientSize", { 3, 5, 7 }), combine(framework::dataset::make("BlockSize", { 3, 5, 7 }), datasets::BorderModes()));
const auto data_precommit = combine(framework::dataset::make("GradientSize", { 3 }), combine(framework::dataset::make("BlockSize", { 3 }), datasets::BorderModes()));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(HarrisCorners)

template <typename T>
using CLHarrisCornersFixture = HarrisCornersValidationFixture<CLTensor, CLAccessor, CLKeyPointArray, CLHarrisCorners, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLHarrisCornersFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SmallImageFiles(), data_precommit), framework::dataset::make("Format",
                                                                                                           Format::U8)))
{
    // Validate output
    CLArrayAccessor<KeyPoint> array(_target);
    validate_keypoints(array.buffer(), array.buffer() + array.num_values(), _reference.begin(), _reference.end(), RelativeTolerance<float>(0.0001f));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLHarrisCornersFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeImageFiles(), data_nightly), framework::dataset::make("Format",
                                                                                                           Format::U8)))
{
    // Validate output
    CLArrayAccessor<KeyPoint> array(_target);
    validate_keypoints(array.buffer(), array.buffer() + array.num_values(), _reference.begin(), _reference.end(), RelativeTolerance<float>(0.0001f));
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
