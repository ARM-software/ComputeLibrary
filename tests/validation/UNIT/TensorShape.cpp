/*
 * Copyright (c) 2017, 2024 Arm Limited.
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
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(TensorShapeValidation)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Construction, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("TensorShape", {
               TensorShape{},
               TensorShape{ 1U },
               TensorShape{ 2U },
               TensorShape{ 2U, 3U },
               TensorShape{ 2U, 3U, 5U },
               TensorShape{ 2U, 3U, 5U, 7U },
               TensorShape{ 2U, 3U, 5U, 7U, 11U },
               TensorShape{ 2U, 3U, 5U, 7U, 11U, 13U }}),
               framework::dataset::make("NumDimensions", { 0U, 1U, 1U, 2U, 3U, 4U, 5U, 6U })),
               framework::dataset::make("TotalSize", { 0U, 1U, 2U, 6U, 30U, 210U, 2310U, 30030U })),
               shape, num_dimensions, total_size)
{
    ARM_COMPUTE_EXPECT(shape.num_dimensions() == num_dimensions, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(shape.total_size() == total_size, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(SetEmpty, framework::DatasetMode::ALL, framework::dataset::make("Dimension", { 0U, 1U, 2U, 3U, 4U, 5U }), dimension)
{
    TensorShape shape;

    shape.set(dimension, 10);

    ARM_COMPUTE_EXPECT(shape.num_dimensions() == dimension + 1, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(shape.total_size() == 10, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(DynamicDimentions, framework::DatasetMode::ALL, zip(
            framework::dataset::make("TensorShape", {
                TensorShape{ 0U },
                TensorShape{ 1U, 0U, 3U },
                TensorShape{ 1U, 2U, 0U },
                TensorShape{ 1U, 2U, 0U, 1U },
                TensorShape{ 0U, 2U, 0U, 1U },
                TensorShape{ 0U, 2U, 0U, 1U },
                TensorShape{ 0U, 0U },
                TensorShape{ 0U, 0U }}),
            framework::dataset::make("dim", {0U, 1U, 2U, 2U, 0U, 2U, 0U, 1U} )),
            shape, dim)
{
    ARM_COMPUTE_ERROR_ON(dim >= TensorShape::num_max_dimensions);
    ARM_COMPUTE_EXPECT(shape.is_dynamic(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(shape.is_dynamic(dim), framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(StaticShapes, framework::DatasetMode::ALL,
            framework::dataset::make("TensorShape", {
                TensorShape{ 1U },
                TensorShape{ 1U, 5U },
                TensorShape{ 1U, 2U, 3U }}),
            shape)
{
    ARM_COMPUTE_EXPECT(!shape.is_dynamic(), framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(StaticDimentions, framework::DatasetMode::ALL, zip(
            framework::dataset::make("TensorShape", {
                TensorShape{ 1U },
                TensorShape{ 1U, 0U },
                TensorShape{ 0U, 2U },
                TensorShape{ 1U, 0U, 3U },
                TensorShape{ 1U, 2U, 0U },
                TensorShape{ 1U, 2U, 3U }}),
            framework::dataset::make("dim", {0U, 0U, 1U, 0U, 1U, 2U})),
            shape, dim)
{
    ARM_COMPUTE_ERROR_ON(dim >= TensorShape::num_max_dimensions);
    ARM_COMPUTE_EXPECT(!shape.is_dynamic(dim), framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // TensorShapeValidation
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
