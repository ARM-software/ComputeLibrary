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

template <typename T>
using NENonLinearFilterFixture = NonLinearFilterValidationFixture<Tensor, Accessor, NENonLinearFilter, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NENonLinearFilterFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::SmallShapes(),
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
