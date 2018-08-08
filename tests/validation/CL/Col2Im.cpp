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
#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"
#include "arm_compute/core/Types.h"
#include "tests/CL/Helper.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/Col2ImLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/Col2ImFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(Col2Im)

using CLCol2Im = CLSynthetizeFunction<CLCol2ImKernel>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::S64),    // Unsupported data type
                                                       TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32),    // Mismatching data type
                                                       TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32),    // Invalid output shape
                                                       TensorInfo(TensorShape(3U, 12U, 4U, 2U), 1, DataType::F32),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 3U, 10U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(3U, 4U, 12U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("ConvolvedWidth", { 3, 3, 3, 3 })),
               framework::dataset::make("ConvolvedHeight", { 4, 4, 4, 4 })),
               framework::dataset::make("NumGroups", { 1, 1, 1, 4 })),
               framework::dataset::make("Expected", { false, false, false, true })),
               input_info, output_info, convolved_width, convolved_height, num_groups, expected)
{
    bool status = bool(CLCol2Im::validate(&input_info, &output_info, std::make_pair(convolved_width, convolved_height), num_groups));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLCol2ImFixture = Col2ImValidationFixture<CLTensor, CLAccessor, CLCol2Im, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLCol2ImFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGroupedCol2ImLayerDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLCol2ImFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGroupedCol2ImLayerDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLCol2ImFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGroupedCol2ImLayerDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLCol2ImFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGroupedCol2ImLayerDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLCol2ImFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGroupedCol2ImLayerDataset(), framework::dataset::make("DataType",
                                                                                                      DataType::QASYMM8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLCol2ImFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGroupedCol2ImLayerDataset(), framework::dataset::make("DataType",
                                                                                                    DataType::QASYMM8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
