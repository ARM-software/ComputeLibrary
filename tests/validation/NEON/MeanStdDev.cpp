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
#include "arm_compute/runtime/NEON/functions/NEMeanStdDev.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/MeanStdDevFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_rel_high_error(0.05f);
RelativeTolerance<float> tolerance_rel_low_error(0.0005f);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(MeanStdDev)

template <typename T>
using NEMeanStdDevFixture = MeanStdDevValidationFixture<Tensor, Accessor, NEMeanStdDev, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEMeanStdDevFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                          DataType::U8)))
{
    // Validate mean output
    validate(_target.first, _reference.first);

    // Validate std_dev output
    validate(_target.second, _reference.second, tolerance_rel_high_error);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEMeanStdDevFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("DataType",
                                                                                                        DataType::U8)))
{
    // Validate mean output
    validate(_target.first, _reference.first, tolerance_rel_low_error);

    // Validate std_dev output
    validate(_target.second, _reference.second, tolerance_rel_high_error);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
