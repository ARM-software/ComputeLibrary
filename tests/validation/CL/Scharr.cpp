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
#include "arm_compute/runtime/CL/functions/CLScharr3x3.h"
#include "tests/CL/CLAccessor.h"
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
TEST_SUITE(CL)
TEST_SUITE(Scharr)

TEST_SUITE(W3x3)
using CLScharr3x3Fixture = ScharrValidationFixture<CLTensor, CLAccessor, CLScharr3x3, uint8_t, int16_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLScharr3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Small2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                      Format::U8)),
                                                                                              datasets::GradientDimensions()))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(CLAccessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(CLAccessor(_target.second), _reference.second, valid_region_y);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLScharr3x3Fixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(), datasets::BorderModes()), framework::dataset::make("Format",
                                                                                                      Format::U8)),
                                                                                              datasets::GradientDimensions()))
{
    // Validate output
    ValidRegion valid_region_x = shape_to_valid_region(_reference.first.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(CLAccessor(_target.first), _reference.first, valid_region_x);

    ValidRegion valid_region_y = shape_to_valid_region(_reference.second.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(1));
    validate(CLAccessor(_target.second), _reference.second, valid_region_y);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
