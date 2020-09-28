/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLSlice.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/SliceOperationsDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SliceOperationsFixtures.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(Slice)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 2U, 5U, 3U), 1, DataType::F32), // Invalid input shape
                                                TensorInfo(TensorShape(27U, 3U, 2U), 1, DataType::F32),         // Negative begin
                                                TensorInfo(TensorShape(27U, 3U, 2U), 1, DataType::F32),         // Big number of coordinates
                                                TensorInfo(TensorShape(27U, 3U, 2U), 1, DataType::F32)
        }),
        framework::dataset::make("Starts", { Coordinates(3, 1, 0), Coordinates(-3, 1, 0), Coordinates(3, 1, 0), Coordinates(3, 1, 0) })),
        framework::dataset::make("Ends", { Coordinates(13, 3, 0),  Coordinates(13, 3, 1), Coordinates(13, 3, 1, 1), Coordinates(13, 3, 1) })),
        framework::dataset::make("Expected", { false, false, false, true })),
        input_info, starts, ends, expected)
{
    TensorInfo output_info;
    const Status status = CLSlice::validate(&input_info.clone()->set_is_resizable(false), &output_info, starts, ends);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLSliceFixture = SliceFixture<CLTensor, CLAccessor, CLSlice, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLSliceFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallSliceDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLSliceFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeSliceDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLSliceFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallSliceDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLSliceFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeSliceDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // Slice
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
