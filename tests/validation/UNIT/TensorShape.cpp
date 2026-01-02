/*
 * Copyright (c) 2017, 2024-2026 Arm Limited.
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
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;
using framework::dataset::zip;
TEST_SUITE(UNIT)
TEST_SUITE(TensorShapeValidation)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Construction, framework::DatasetMode::ALL, zip(
               make("TensorShape", {
               TensorShape{},
               TensorShape{ 1U },
               TensorShape{ 2U },
               TensorShape{ 2U, 3U },
               TensorShape{ 2U, 3U, 5U },
               TensorShape{ 2U, 3U, 5U, 7U },
               TensorShape{ 2U, 3U, 5U, 7U, 11U },
               TensorShape{ 2U, 3U, 5U, 7U, 11U, 13U }}),
               make("NumDimensions", { 0U, 1U, 1U, 2U, 3U, 4U, 5U, 6U }),
               make("TotalSize", { 0U, 1U, 2U, 6U, 30U, 210U, 2310U, 30030U })),
               shape, num_dimensions, total_size)
{
    ARM_COMPUTE_EXPECT(shape.num_dimensions() == num_dimensions, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(shape.total_size() == total_size, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(SetEmpty, framework::DatasetMode::ALL, make("Dimension", {0U, 1U, 2U, 3U, 4U, 5U}), dimension)
{
    TensorShape shape;

    shape.set(dimension, 10);

    ARM_COMPUTE_EXPECT(shape.num_dimensions() == dimension + 1, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(shape.total_size() == 10, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(TotalSize,
               framework::DatasetMode::ALL,
               zip(make("TensorShape",
                        {TensorShape{200000U, 300000U}, TensorShape{2000U, 3000U, 5000U}, TensorShape{50000000000ULL}}),
                   make("TotalSize", {size_t(60000000000ULL), size_t(30000000000ULL), size_t(50000000000ULL)})),
               shape,
               total_size)
{
    if (sizeof(size_t) >= 8UL)
    {
        ARM_COMPUTE_EXPECT(shape.total_size() == total_size, framework::LogLevel::ERRORS);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Test SKIPPED as size_t is less than 8 bytes in this system.");
    }
}

TEST_CASE(TotalSizeLowerUpperSmallShape, framework::DatasetMode::ALL)
{
    const auto tensor_shape = TensorShape(2U, 3U, 5U, 6U, 7U);

    // Lower
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(0) == 1U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(1) == 2U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(2) == 6U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(3) == 30U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(4) == 180U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(5) == 1260U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(6) == 1260U, framework::LogLevel::ERRORS);

    // Upper
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_upper(0) == 1260U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_upper(1) == 630U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_upper(2) == 210U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_upper(3) == 42U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_upper(4) == 7U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor_shape.total_size_upper(5) == 1U, framework::LogLevel::ERRORS);
}

TEST_CASE(TotalSizeLowerUpperBigShape, framework::DatasetMode::ALL)
{
    if (sizeof(size_t) >= 8UL)
    {
        const auto tensor_shape = TensorShape(200U, 300U, 500U, 600U, 700U);
        ARM_COMPUTE_EXPECT(tensor_shape.total_size() == 12600000000000ULL, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(tensor_shape.total_size_lower(4) == 18000000000ULL, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(tensor_shape.total_size_upper(1) == 63000000000ULL, framework::LogLevel::ERRORS);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Test SKIPPED as size_t is less than 8 bytes in this system.");
    }
}

DATA_TEST_CASE(TotalSizeBigShape,
               framework::DatasetMode::ALL,
               zip(make("TensorShape",
                        {TensorShape{200000U, 300000U}, TensorShape{2000U, 3000U, 5000U}, TensorShape{50000000000ULL}}),
                   make("TotalSize", {size_t(60000000000ULL), size_t(30000000000ULL), size_t(50000000000ULL)})),
               shape,
               total_size)
{
    if (sizeof(size_t) >= 8UL)
    {
        ARM_COMPUTE_EXPECT(shape.total_size() == total_size, framework::LogLevel::ERRORS);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Test SKIPPED as size_t is less than 8 bytes in this system.");
    }
}

NON_CONST_DATA_TEST_CASE(Collapse,
                         framework::DatasetMode::ALL,
                         zip(make("TensorShape",
                                  {
                                      TensorShape(2U, 3U, 5U, 6U, 7U),
                                      TensorShape(2U, 3U, 5U, 6U, 7U),
                                      TensorShape(2U, 3U, 5U, 6U, 7U),
                                      TensorShape(2U, 3U, 5U, 6U, 7U),
                                      TensorShape(2U, 3U, 5U, 6U, 7U),
                                      TensorShape(2U, 3U, 5U, 6U, 7U),
                                  }),
                             make("NumDimensionsToCollapse", {2, 2, 1, 3, 5, 3}),
                             make("FirstDimensionToCollapseFrom",
                                  // If negative, we'll ignore and let the call use the default value.
                                  {-1, 0, 3, 1, 0, 3}),
                             make("Expected",
                                  {TensorShape(6U, 5U, 6U, 7U), TensorShape(6U, 5U, 6U, 7U),
                                   TensorShape(2U, 3U, 5U, 6U, 7U), TensorShape(2U, 90U, 7U), TensorShape(1260U),
                                   TensorShape(2U, 3U, 5U, 42U)})),
                         tensor_shape,
                         num_dims,
                         first_dim,
                         expected)
{
    if (first_dim >= 0)
    {
        tensor_shape.collapse(num_dims, first_dim);
    }
    else
    {
        tensor_shape.collapse(num_dims);
    }

    ARM_COMPUTE_EXPECT(tensor_shape == expected, framework::LogLevel::ERRORS);
}

TEST_CASE(CollapseBigShape, framework::DatasetMode::ALL)
{
    if (sizeof(size_t) >= 8UL)
    {
        auto shape1 = TensorShape(2000U, 3000U, 5000U, 6000U, 3U, 4U);
        shape1.collapse(4);

        ARM_COMPUTE_EXPECT(shape1 == TensorShape(180000000000000ULL, 3ULL, 4ULL), framework::LogLevel::ERRORS);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Test SKIPPED as size_t is less than 8 bytes in this system.");
    }
}

DATA_TEST_CASE(CollapsedFrom,
               framework::DatasetMode::ALL,
               zip(make("TensorShape",
                        {
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                        }),
                   make("FirstDimensionToCollapseFrom", {0, 1, 2, 3, 4, 5}),
                   make("Expected",
                        {
                            TensorShape(1260U),
                            TensorShape(2U, 630U),
                            TensorShape(2U, 3U, 210U),
                            TensorShape(2U, 3U, 5U, 42U),
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                            TensorShape(2U, 3U, 5U, 6U, 7U),
                        })),
               tensor_shape,
               first_dim,
               expected)
{
    ARM_COMPUTE_EXPECT(tensor_shape.collapsed_from(first_dim) == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // TensorShapeValidation
TEST_SUITE_END() // UNIT
} // namespace validation
} // namespace test
} // namespace arm_compute
