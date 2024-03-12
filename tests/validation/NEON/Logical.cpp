/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NELogical.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/LogicalFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)

TEST_SUITE(LogicalAnd)
template <typename T>
using NELogicalAndFixture = LogicalAndValidationFixture<Tensor, Accessor, NELogicalAnd, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NELogicalAndFixture<uint8_t>, framework::DatasetMode::ALL, zip(datasets::SmallShapes(), datasets::SmallShapes()))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NELogicalAndFixture<uint8_t>, framework::DatasetMode::ALL, datasets::SmallShapesBroadcast())
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // LogicalAnd

TEST_SUITE(LogicalOr)
template <typename T>
using NELogicalOrFixture = LogicalOrValidationFixture<Tensor, Accessor, NELogicalOr, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NELogicalOrFixture<uint8_t>, framework::DatasetMode::ALL, zip(datasets::SmallShapes(), datasets::SmallShapes()))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NELogicalOrFixture<uint8_t>, framework::DatasetMode::ALL, datasets::SmallShapesBroadcast())
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // LogicalOr

TEST_SUITE(LogicalNot)

template <typename T>
using NELogicalNotFixture = LogicalNotValidationFixture<Tensor, Accessor, NELogicalNot, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NELogicalNotFixture<uint8_t>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                    DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // LogicalNot
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
