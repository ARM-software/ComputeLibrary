/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGaussianPyramid.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GaussianPyramidHalfFixture.h"
#include "tests/validation/reference/Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_fp32(1.0f); /**< Tolerance value for comparing reference's output against implementation's output */

const auto small_gaussian_pyramid_levels = combine(datasets::Medium2DShapes(), datasets::BorderModes()) * framework::dataset::make("numlevels", 2, 3);
const auto large_gaussian_pyramid_levels = combine(datasets::Large2DShapes(), datasets::BorderModes()) * framework::dataset::make("numlevels", 2, 5);

template <typename T, typename U>
inline void validate_gaussian_pyramid(const Pyramid &target, const std::vector<SimpleTensor<T>> &reference, BorderMode border_mode, U tolerance)
{
    ValidRegion prev_valid_region = shape_to_valid_region(reference[0].shape());

    for(size_t i = 1; i < reference.size(); ++i)
    {
        const ValidRegion valid_region = shape_to_valid_region_gaussian_pyramid_half(reference[i - 1].shape(), prev_valid_region, (border_mode == BorderMode::UNDEFINED));

        // Validate outputs
        validate(Accessor(*(target.get_pyramid_level(i))), reference[i], valid_region, tolerance);

        // Keep the valid region for the next level
        prev_valid_region = valid_region;
    }
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(GaussianPyramid)
TEST_SUITE(Half)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, large_gaussian_pyramid_levels,
               shape, border_mode, num_levels)
{
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);

    // Create pyramid
    PyramidInfo pyramid_info(num_levels, SCALE_PYRAMID_HALF, shape, Format::U8);
    Pyramid     dst;
    dst.init(pyramid_info);

    NEGaussianPyramidHalf gaussian_pyramid_half;
    gaussian_pyramid_half.configure(&src, &dst, border_mode, 0);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

    for(size_t level = 0; level < pyramid_info.num_levels(); ++level)
    {
        ARM_COMPUTE_EXPECT(dst.get_pyramid_level(level)->info()->is_resizable(), framework::LogLevel::ERRORS);
    }
}

template <typename T>
using NEGaussianPyramidHalfFixture = GaussianPyramidHalfValidationFixture<Tensor, Accessor, NEGaussianPyramidHalf, T, Pyramid>;

FIXTURE_DATA_TEST_CASE(RunSmallGaussianPyramidHalf, NEGaussianPyramidHalfFixture<uint8_t>, framework::DatasetMode::ALL, small_gaussian_pyramid_levels)
{
    validate_gaussian_pyramid(_target, _reference, _border_mode, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunLargeGaussianPyramidHalf, NEGaussianPyramidHalfFixture<uint8_t>, framework::DatasetMode::NIGHTLY, large_gaussian_pyramid_levels)
{
    validate_gaussian_pyramid(_target, _reference, _border_mode, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
