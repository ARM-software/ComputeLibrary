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

TEST_SUITE(X)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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

TEST_SUITE(X)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel5x5Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel5x5Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel5x5Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
TEST_SUITE(X)
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel7x7Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel7x7Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
FIXTURE_DATA_TEST_CASE(RunSmall, NESobel7x7Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
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
