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
#include "arm_compute/runtime/NEON/functions/NESobel3x3.h"
#include "arm_compute/runtime/NEON/functions/NESobel5x5.h"
#include "arm_compute/runtime/NEON/functions/NESobel7x7.h"
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
#include "tests/validation/fixtures/SobelFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Sobel)

TEST_SUITE(W3x3)
using NESobel3x3Fixture = SobelValidationFixture<Tensor, Accessor, NESobel3x3, uint8_t, int16_t>;

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

    // Create sobel 3x3 configure function
    NESobel3x3 sobel;
    sobel.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);

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

TEST_SUITE(X)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel3x3Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_X)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_X)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);
}
TEST_SUITE_END()
TEST_SUITE(Y)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel3x3Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_Y)))
{
    // Validate output
    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_Y)))
{
    // Validate output
    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()
TEST_SUITE(XY)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel3x3Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_XY)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_XY)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(W5x5)
using NESobel5x5Fixture = SobelValidationFixture<Tensor, Accessor, NESobel5x5, uint8_t, int16_t>;

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

    // Create sobel 5x5 configure function
    NESobel5x5 sobel;
    sobel.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);

    // Validate valid region
    constexpr BorderSize border_size{ 2 };
    const ValidRegion    dst_valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size);

    validate(dst_x.info()->valid_region(), dst_valid_region);
    validate(dst_y.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 16);

    calculator.set_border_mode(border_mode);
    calculator.set_border_size(2);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_processed_elements(8);
    calculator.set_access_offset(-2);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst_x.info()->padding(), dst_padding);
    validate(dst_y.info()->padding(), dst_padding);
}
TEST_SUITE(X)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel5x5Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_X)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.first), _reference.first, valid_region_x);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel5x5Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_X)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.first), _reference.first, valid_region_x);
}
TEST_SUITE_END()
TEST_SUITE(Y)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel5x5Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_Y)))
{
    // Validate output
    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel5x5Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_Y)))
{
    // Validate output
    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()
TEST_SUITE(XY)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel5x5Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_XY)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel5x5Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_XY)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(2));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(W7x7)
using NESobel7x7Fixture = SobelValidationFixture<Tensor, Accessor, NESobel7x7, uint8_t, int32_t>;

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
    Tensor dst_x = create_tensor<Tensor>(shape, DataType::S32);
    Tensor dst_y = create_tensor<Tensor>(shape, DataType::S32);

    src.info()->set_format(format);
    dst_x.info()->set_format(Format::S32);
    dst_y.info()->set_format(Format::S32);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst_x.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst_y.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create sobel 7x7 configure function
    NESobel7x7 sobel;
    sobel.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);

    // Validate valid region
    constexpr BorderSize border_size{ 3 };
    const ValidRegion    dst_valid_region = shape_to_valid_region(shape, border_mode == BorderMode::UNDEFINED, border_size);

    validate(dst_x.info()->valid_region(), dst_valid_region);
    validate(dst_y.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);

    calculator.set_border_mode(border_mode);
    calculator.set_border_size(3);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-3);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst_x.info()->padding(), dst_padding);
    validate(dst_y.info()->padding(), dst_padding);
}
TEST_SUITE(X)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel7x7Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_X)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.first), _reference.first, valid_region_x);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel7x7Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_X)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.first), _reference.first, valid_region_x);
}
TEST_SUITE_END()
TEST_SUITE(Y)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel7x7Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_Y)))
{
    // Validate output
    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel7x7Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_Y)))
{
    // Validate output
    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()
TEST_SUITE(XY)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel7x7Fixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               framework::dataset::make("GradientDimension", GradientDimension::GRAD_XY)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NESobel7x7Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                     Format::U8)),
                                                                                             framework::dataset::make("GradientDimension", GradientDimension::GRAD_XY)))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(3));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
