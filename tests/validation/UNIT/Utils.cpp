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
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "utils/TypePrinter.h"

#include <stdexcept>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

TEST_SUITE(UNIT)
TEST_SUITE(Utils)

DATA_TEST_CASE(RoundHalfUp, framework::DatasetMode::ALL, zip(framework::dataset::make("FloatIn", { 1.f, 1.2f, 1.5f, 2.5f, 2.9f, -3.f, -3.5f, -3.8f, -4.3f, -4.5f }),
                                                             framework::dataset::make("FloatOut", { 1.f, 1.f, 2.f, 3.f, 3.f, -3.f, -3.f, -4.f, -4.f, -4.f })),
               value, result)
{
    ARM_COMPUTE_EXPECT(round_half_up(value) == result, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(RoundHalfEven, framework::DatasetMode::ALL, zip(framework::dataset::make("FloatIn", { 1.f, 1.2f, 1.5f, 2.5f, 2.9f, -3.f, -3.5f, -3.8f, -4.3f, -4.5f }),
                                                               framework::dataset::make("FloatOut", { 1.f, 1.f, 2.f, 2.f, 3.f, -3.f, -4.f, -4.f, -4.f, -4.f })),
               value, result)
{
    ARM_COMPUTE_EXPECT(round_half_even(value) == result, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(Index2Coord, framework::DatasetMode::ALL, zip(zip(framework::dataset::make("Shape", { TensorShape{ 1U }, TensorShape{ 2U }, TensorShape{ 2U, 3U } }), framework::dataset::make("Index", { 0, 1, 2 })),
                                                             framework::dataset::make("Coordinates", { Coordinates{ 0 }, Coordinates{ 1 }, Coordinates{ 0, 1 } })),
               shape, index, ref_coordinate)
{
    Coordinates coordinate = index2coord(shape, index);

    ARM_COMPUTE_EXPECT(compare_dimensions(coordinate, ref_coordinate), framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(RoundFloatToZero, framework::DatasetMode::ALL, zip(framework::dataset::make("FloatIn", { 1.f, 1.2f, 1.5f, 2.5f, 2.9f, -3.f, -3.5f, -3.8f, -4.3f, -4.5f }),
                                                                  framework::dataset::make("FloatOut", { 1.f, 1.f, 1.f, 2.f, 2.f, -3.f, -3.f, -3.f, -4.f, -4.f })),
               value, result)
{
    ARM_COMPUTE_EXPECT(round(value, RoundingPolicy::TO_ZERO) == result, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(RoundFloatToNearestUp, framework::DatasetMode::ALL, zip(framework::dataset::make("FloatIn", { 1.f, 1.2f, 1.5f, 2.5f, 2.9f, -3.f, -3.5f, -3.8f, -4.3f, -4.5f }),
                                                                       framework::dataset::make("FloatOut", { 1.f, 1.f, 2.f, 3.f, 3.f, -3.f, -4.f, -4.f, -4.f, -5.f })),
               value, result)
{
    ARM_COMPUTE_EXPECT(round(value, RoundingPolicy::TO_NEAREST_UP) == result, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(Coord2Index, framework::DatasetMode::ALL, zip(zip(framework::dataset::make("Shape", { TensorShape{ 1U }, TensorShape{ 2U }, TensorShape{ 2U, 3U } }),
                                                                 framework::dataset::make("Coordinates", { Coordinates{ 0 }, Coordinates{ 1 }, Coordinates{ 0, 1 } })),
                                                             framework::dataset::make("Index", { 0, 1, 2 })),
               shape, coordinate, ref_index)
{
    int index = coord2index(shape, coordinate);

    ARM_COMPUTE_EXPECT(index == ref_index, framework::LogLevel::ERRORS);
}

TEST_SUITE_END()
TEST_SUITE_END()
