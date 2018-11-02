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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEFastCorners.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/ArrayAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ImageFileDatasets.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FastCornersFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/* Radius of the Bresenham circle around the candidate point */
const unsigned int bresenham_radius = 3;
/* Tolerance used to compare corner strengths */
const AbsoluteTolerance<float> tolerance(0.5f);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(FastCorners)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(concat(datasets::Small2DShapes(), datasets::Large2DShapes()),
                                                                                   framework::dataset::make("Format", Format::U8)),
                                                                           framework::dataset::make("SuppressNonMax", { false, true })),
                                                                   framework::dataset::make("BorderMode", BorderMode::UNDEFINED)),
               shape, format, suppress_nonmax, border_mode)
{
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> int_dist(0, 255);
    std::uniform_real_distribution<float>  real_dist(0, 255);

    const uint8_t constant_border_value = int_dist(gen);
    const float   threshold             = real_dist(gen);

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type_from_format(format));
    src.info()->set_format(format);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

    KeyPointArray corners;

    // Create and configure function
    NEFastCorners fast_corners;
    fast_corners.configure(&src, threshold, suppress_nonmax, &corners, border_mode, constant_border_value);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 1); // elems_processed

    calculator.set_border_size(bresenham_radius);
    calculator.set_access_offset(-bresenham_radius);
    calculator.set_accessed_elements(8); // elems_read

    validate(src.info()->padding(), calculator.required_padding());
}

template <typename T>
using NEFastCornersFixture = FastCornersValidationFixture<Tensor, Accessor, KeyPointArray, NEFastCorners, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEFastCornersFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallImageFiles(), framework::dataset::make("Format", Format::U8)),
                                                                                                                   framework::dataset::make("SuppressNonMax", { false, true })),
                                                                                                           framework::dataset::make("BorderMode", BorderMode::UNDEFINED)))
{
    // Validate output
    ArrayAccessor<KeyPoint> array(_target);
    validate_keypoints(array.buffer(), array.buffer() + array.num_values(), _reference.begin(), _reference.end(), tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFastCornersFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeImageFiles(), framework::dataset::make("Format", Format::U8)),
                                                                                                                 framework::dataset::make("SuppressNonMax", { false, true })),
                                                                                                         framework::dataset::make("BorderMode", BorderMode::UNDEFINED)))
{
    // Validate output
    ArrayAccessor<KeyPoint> array(_target);
    validate_keypoints(array.buffer(), array.buffer() + array.num_values(), _reference.begin(), _reference.end(), tolerance);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
