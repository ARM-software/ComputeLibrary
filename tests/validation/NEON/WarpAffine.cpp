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
#include "arm_compute/runtime/NEON/functions/NEWarpAffine.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/AssetsLibrary.h"
#include "tests/NEON/Accessor.h"
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

TEST_SUITE(NEON)
TEST_SUITE(WarpAffine)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::U8)),
                                                                           framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                   datasets::BorderModes()),
               shape, data_type, policy, border_mode)
{
    // Generate a random constant value if border_mode is constant
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    uint8_t                                constant_border_value = distribution_u8(gen);

    // Create the matrix
    std::array<float, 9> matrix{ {} };
    fill_warp_matrix<9>(matrix);

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEWarpAffine warp_affine;
    warp_affine.configure(&src, &dst, matrix, policy, border_mode, constant_border_value);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);

    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 1);
    calculator.set_border_mode(border_mode);
    calculator.set_border_size(1);

    const PaddingSize read_padding(1);
    const PaddingSize write_padding = calculator.required_padding();

    validate(src.info()->padding(), read_padding);
    validate(dst.info()->padding(), write_padding);
}

template <typename T>
using NEWarpAffineFixture = WarpAffineValidationFixture<Tensor, Accessor, NEWarpAffine, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEWarpAffineFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::U8)),
                                                                                                                  framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                          datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, _valid_mask, tolerance, 0.02f);
}
DISABLED_FIXTURE_DATA_TEST_CASE(RunLarge, NEWarpAffineFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                                 datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, _valid_mask, tolerance, 0.02f);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
