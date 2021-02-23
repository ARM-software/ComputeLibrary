/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#include "arm_compute/runtime/NEON/functions/NEDerivative.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/GradientDimensionDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DerivativeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Derivative)

using NEDerivativeFixture = DerivativeValidationFixture<Tensor, Accessor, NEDerivative, uint8_t, int16_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEDerivativeFixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               datasets::GradientDimensions()))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDerivativeFixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                       Format::U8)),
                                                                                               datasets::GradientDimensions()))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(Accessor(_target.second), _reference.second, valid_region_y);
}

TEST_SUITE_END() // Derivative
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
