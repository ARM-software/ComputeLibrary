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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEAbsoluteDifference.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/AbsoluteDifferenceFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data sets **/
const auto AbsoluteDifferenceU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)), framework::dataset::make("DataType",
                                                 DataType::U8));
const auto AbsoluteDifferenceS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::U8, DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                                  framework::dataset::make("DataType", DataType::S16));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(AbsoluteDifference)

template <typename T>
using NEAbsoluteDifferenceFixture = AbsoluteDifferenceValidationFixture<Tensor, Accessor, NEAbsoluteDifference, T>;

TEST_SUITE(U8)

FIXTURE_DATA_TEST_CASE(RunSmall, NEAbsoluteDifferenceFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::SmallShapes(), AbsoluteDifferenceU8Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEAbsoluteDifferenceFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AbsoluteDifferenceU8Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)

FIXTURE_DATA_TEST_CASE(RunSmall, NEAbsoluteDifferenceFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(datasets::SmallShapes(), AbsoluteDifferenceS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEAbsoluteDifferenceFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AbsoluteDifferenceS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE_END() // AbsoluteDifference
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
