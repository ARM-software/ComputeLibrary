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

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::Small2DShapes(), data_nightly), framework::dataset::make("Format", Format::U8)), shape,
               gradient_size, block_size, border_mode, format)
{
    std::mt19937                          gen(library->seed());
    std::uniform_real_distribution<float> real_dist(0.f, 0.01f);

    const float threshold   = real_dist(gen);
    const float sensitivity = real_dist(gen);

    constexpr float max_euclidean_distance = 30.f;
    real_dist                              = std::uniform_real_distribution<float>(0.f, max_euclidean_distance);
    const float min_dist                   = real_dist(gen);

    // Generate a random constant value
    std::uniform_int_distribution<uint8_t> int_dist(0, 255);
    const uint8_t                          constant_border_value = int_dist(gen);

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type_from_format(format));
    src.info()->set_format(format);
    CLKeyPointArray corners;

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create harris corners configure function
    CLHarrisCorners harris_corners;
    harris_corners.configure(&src, threshold, min_dist, sensitivity, gradient_size, block_size, &corners, border_mode, constant_border_value);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);

    calculator.set_border_mode(border_mode);
    calculator.set_border_size(gradient_size / 2);
    calculator.set_access_offset(-gradient_size / 2);
    calculator.set_accessed_elements(16);

    const PaddingSize padding = calculator.required_padding();

    validate(src.info()->padding(), padding);
}

template <typename T>
using CLHarrisCornersFixture = HarrisCornersValidationFixture<CLTensor, CLAccessor, CLKeyPointArray, CLHarrisCorners, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLHarrisCornersFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallImageFiles(), data_precommit), framework::dataset::make("Format",
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
