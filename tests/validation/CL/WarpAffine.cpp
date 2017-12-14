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
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/functions/CLWarpAffine.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/InterpolationPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WarpAffineFixture.h"
#include "tests/validation/reference/Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance */
constexpr AbsoluteTolerance<uint8_t> tolerance(1);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(WarpAffine)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                           framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                   datasets::BorderModes()),
               shape, data_type, policy, border_mode)
{
    // Generate a random constant value if border_mode is constant
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    uint8_t                                constant_border_value = distribution_u8(gen);

    // Create the matrix
    std::array<float, 6> matrix{ {} };
    fill_warp_matrix<6>(matrix);

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWarpAffine warp_affine;
    warp_affine.configure(&src, &dst, matrix.data(), policy, border_mode, constant_border_value);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);

    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    int               total_right  = ceil_to_multiple(shape[0], 4);
    const int         access_right = total_right + (((total_right - shape[0]) == 0) ? 1 : 0);
    const PaddingSize read_padding(1, access_right - shape[0], 1, 1);
    validate(src.info()->padding(), read_padding);

    PaddingCalculator calculator(shape.x(), 4);
    validate(dst.info()->padding(), calculator.required_padding());
}

template <typename T>
using CLWarpAffineFixture = WarpAffineValidationFixture<CLTensor, CLAccessor, CLWarpAffine, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLWarpAffineFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::U8)),
                                                                                                                  framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                          datasets::BorderModes()))
{
    // Validate output
    validate(CLAccessor(_target), _reference, _valid_mask, tolerance, 0.02f);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWarpAffineFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::U8)),
                                                                                                                framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                        datasets::BorderModes()))
{
    // Validate output
    validate(CLAccessor(_target), _reference, _valid_mask, tolerance, 0.02f);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
