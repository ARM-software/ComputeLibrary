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
#include "arm_compute/runtime/NEON/functions/NENonLinearFilter.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/MatrixPatternDataset.h"
#include "tests/datasets/NonLinearFilterFunctionDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/NonLinearFilterFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(NonLinearFilter)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallShapes(), datasets::NonLinearFilterFunctions()),
                                                                                   framework::dataset::make("MaskSize", { 3U, 5U })),
                                                                           datasets::MatrixPatterns()),
                                                                   datasets::BorderModes()),
               shape, function, mask_size, pattern, border_mode)
{
    std::mt19937                           generator(library->seed());
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    const uint8_t                          constant_border_value = distribution_u8(generator);

    // Create the mask
    uint8_t mask[mask_size * mask_size];
    fill_mask_from_pattern(mask, mask_size, mask_size, pattern);
    const auto half_mask_size = static_cast<int>(mask_size / 2);

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NENonLinearFilter filter;
    filter.configure(&src, &dst, function, mask_size, pattern, mask, border_mode, constant_border_value);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, BorderSize(half_mask_size));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), ((MatrixPattern::OTHER == pattern) ? 1 : 8));
    calculator.set_border_mode(border_mode);
    calculator.set_border_size(half_mask_size);

    const PaddingSize write_padding = calculator.required_padding(PaddingCalculator::Option::EXCLUDE_BORDER);

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-half_mask_size);

    const PaddingSize read_padding = calculator.required_padding(PaddingCalculator::Option::INCLUDE_BORDER);

    validate(src.info()->padding(), read_padding);
    validate(dst.info()->padding(), write_padding);
}

template <typename T>
using NENonLinearFilterFixture = NonLinearFilterValidationFixture<Tensor, Accessor, NENonLinearFilter, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NENonLinearFilterFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       datasets::NonLinearFilterFunctions()),
                                                                                                                       framework::dataset::make("MaskSize", { 3U, 5U })),
                                                                                                                       datasets::MatrixPatterns()),
                                                                                                                       datasets::BorderModes()),
                                                                                                               framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), _border_size));
}

FIXTURE_DATA_TEST_CASE(RunLarge, NENonLinearFilterFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                     datasets::NonLinearFilterFunctions()),
                                                                                                                     framework::dataset::make("MaskSize", { 3U, 5U })),
                                                                                                                     datasets::MatrixPatterns()),
                                                                                                                     datasets::BorderModes()),
                                                                                                             framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), _border_size));
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
