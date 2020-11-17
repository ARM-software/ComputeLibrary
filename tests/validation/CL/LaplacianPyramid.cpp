/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/CLPyramid.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLLaplacianPyramid.h"
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/LaplacianPyramidFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/* Absolute tolerance value for comparing reference's output against implementation's output for DataType::S16
 * Tolerance is needed for calculation uncertainties introduced from the layers
 */
AbsoluteTolerance<int16_t> tolerance_int16(1);
const auto                 small_laplacian_pyramid_levels = framework::dataset::make("NumLevels", 2, 3);
const auto                 large_laplacian_pyramid_levels = framework::dataset::make("NumLevels", 2, 5);

const auto formats = combine(framework::dataset::make("FormatIn", Format::U8), framework::dataset::make("FormatOut", Format::S16));

template <typename T>
inline void validate_laplacian_pyramid(const CLPyramid &target, const std::vector<SimpleTensor<T>> &reference, BorderMode border_mode)
{
    CLTensor   *level_image  = target.get_pyramid_level(0);
    ValidRegion valid_region = shape_to_valid_region(reference[0].shape(), border_mode == BorderMode::UNDEFINED, BorderSize(2));

    // Validate lowest level
    validate(CLAccessor(*level_image), reference[0], valid_region);

    // Validate remaining levels
    for(size_t lev = 1; lev < target.info()->num_levels(); lev++)
    {
        level_image                = target.get_pyramid_level(lev);
        CLTensor *prev_level_image = target.get_pyramid_level(lev - 1);

        valid_region = shape_to_valid_region_laplacian_pyramid(prev_level_image->info()->tensor_shape(),
                                                               prev_level_image->info()->valid_region(),
                                                               border_mode == BorderMode::UNDEFINED);

        // Validate level
        validate(CLAccessor(*level_image), reference[lev], valid_region, tolerance_int16);
    }
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(LaplacianPyramid)

// *INDENT-OFF*
// clang-format off

using CLLaplacianPyramidFixture = LaplacianPyramidValidationFixture<CLTensor, CLAccessor, CLLaplacianPyramid, uint8_t, int16_t, CLPyramid>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLLaplacianPyramidFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(
                       datasets::Medium2DShapes(),
                       datasets::BorderModes()),
                       small_laplacian_pyramid_levels),
                       formats))
{
    validate_laplacian_pyramid(_target, _reference, _border_mode);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLLaplacianPyramidFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(
                       datasets::Large2DShapes(),
                       datasets::BorderModes()),
                       large_laplacian_pyramid_levels),
                       formats))
{
    validate_laplacian_pyramid(_target, _reference, _border_mode);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
