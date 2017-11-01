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
#include "arm_compute/runtime/NEON/functions/NEScharr3x3.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/GradientDimensionDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ScharrFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Scharr)

TEST_SUITE(W3x3)
using NEScharr3x3Fixture = ScharrValidationFixture<Tensor, Accessor, NEScharr3x3, uint8_t, int16_t>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::Small2DShapes(), datasets::Large2DShapes()), datasets::BorderModes()), framework::dataset::make("Format",
                                                                   Format::U8)),
               shape, border_mode, format)
{
    // Generate a random constant value
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> int_dist(0, 255);
    const uint8_t                          constant_border_value = int_dist(gen);

    // Create tensors
    Tensor src   = create_tensor<Tensor>(shape, data_type_from_format(format));
    Tensor dst_x = create_tensor<Tensor>(shape, DataType::S16);
    Tensor dst_y = create_tensor<Tensor>(shape, DataType::S16);

    src.info()->set_format(format);
    dst_x.info()->set_format(Format::S16);
    dst_y.info()->set_format(Format::S16);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst_x.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst_y.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create scharr 3x3 configure function
    NEScharr3x3 scharr;
    scharr.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);

    // Validate valid region
    constexpr BorderSize border_size{ 1 };
    const ValidRegion    dst_valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size);

    validate(dst_x.info()->valid_region(), dst_valid_region);
    validate(dst_y.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);

    calculator.set_border_mode(border_mode);
    calculator.set_border_size(1);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-1);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst_x.info()->padding(), dst_padding);
    validate(dst_y.info()->padding(), dst_padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEScharr3x3Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                        Format::U8)),
                                                                                                datasets::GradientDimensions()))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEScharr3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                      Format::U8)),
                                                                                              datasets::GradientDimensions()))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
