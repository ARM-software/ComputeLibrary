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
#include "arm_compute/runtime/CL/functions/CLMinMaxLocation.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/CLArrayAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/MinMaxLocationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(MinMaxLocation)

template <typename T>
using CLMinMaxLocationFixture = MinMaxLocationValidationFixture<CLTensor, CLAccessor, CLArray<Coordinates2D>, CLArrayAccessor<Coordinates2D>, CLMinMaxLocation, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMinMaxLocationFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                              DataType::U8)))
{
    validate_min_max_loc(_target, _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLMinMaxLocationFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("DataType",
                                                                                                            DataType::U8)))
{
    validate_min_max_loc(_target, _reference);
}

TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMinMaxLocationFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                              DataType::S16)))
{
    validate_min_max_loc(_target, _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLMinMaxLocationFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("DataType",
                                                                                                            DataType::S16)))
{
    validate_min_max_loc(_target, _reference);
}

TEST_SUITE_END() // S16

TEST_SUITE(Float)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMinMaxLocationFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                            DataType::F32)))
{
    validate_min_max_loc(_target, _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLMinMaxLocationFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("DataType",
                                                                                                          DataType::F32)))
{
    validate_min_max_loc(_target, _reference);
}

TEST_SUITE_END() // F32

TEST_SUITE_END() // MinMaxLocation
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
