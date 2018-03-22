/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLCopy.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CopyFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(Copy)

template <typename T>
using CLCopyFixture = CopyFixture<CLTensor, CLAccessor, CLCopy, T>;

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLCopyFixture<float>, framework::DatasetMode::PRECOMMIT, combine(zip(datasets::SmallShapes(), datasets::SmallShapes()), framework::dataset::make("DataType",
                                                                                                  DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLCopyFixture<float>, framework::DatasetMode::NIGHTLY, combine(zip(datasets::LargeShapes(), datasets::LargeShapes()), framework::dataset::make("DataType",
                                                                                                DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // F32

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLCopyFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(zip(datasets::SmallShapes(), datasets::SmallShapes()), framework::dataset::make("DataType",
                                                                                                    DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLCopyFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(zip(datasets::LargeShapes(), datasets::LargeShapes()), framework::dataset::make("DataType",
                                                                                                  DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLCopyFixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(zip(datasets::SmallShapes(), datasets::SmallShapes()), framework::dataset::make("DataType",
                                                                                                     DataType::U16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLCopyFixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(zip(datasets::LargeShapes(), datasets::LargeShapes()), framework::dataset::make("DataType",
                                                                                                   DataType::U16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE_END() // Copy
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
