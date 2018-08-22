/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ReshapeLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReshapeLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(ReshapeLayer)

// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
                                                              framework::dataset::make("InputInfo",
{
    TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),
    TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32),
    TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32), // mismatching dimensions
    TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F16), // mismatching types
}),
framework::dataset::make("OutputInfo",
{
    TensorInfo(TensorShape(9U, 5U, 21U), 1, DataType::F32),
    TensorInfo(TensorShape(8U, 24U, 4U), 1, DataType::F32),
    TensorInfo(TensorShape(192U, 192U),  1, DataType::F32),
    TensorInfo(TensorShape(9U, 5U, 21U), 1, DataType::F32),
})),
framework::dataset::make("Expected", { true, true, false, false })),
input_info, output_info, expected)
{
    // Create Fully Connected layer info
    Status status = CLReshapeLayer::validate(&input_info.clone()->set_is_resizable(false),
                                             &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

template <typename T>
using CLReshapeLayerFixture = ReshapeLayerValidationFixture<CLTensor, CLAccessor, CLReshapeLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReshapeLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallReshapeLayerDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReshapeLayerFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallReshapeLayerDataset(), framework::dataset::make("DataType",
                                                                                                   DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReshapeLayerFixture<uint8_t>, framework::DatasetMode::ALL, combine(datasets::SmallReshapeLayerDataset(), framework::dataset::make("DataType", { DataType::U8, DataType::QASYMM8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReshapeLayerFixture<int8_t>, framework::DatasetMode::ALL, combine(datasets::SmallReshapeLayerDataset(), framework::dataset::make("DataType", DataType::S8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReshapeLayerFixture<int16_t>, framework::DatasetMode::ALL, combine(datasets::SmallReshapeLayerDataset(), framework::dataset::make("DataType", DataType::S16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
