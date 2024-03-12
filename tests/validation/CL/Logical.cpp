/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLLogicalAnd.h"
#include "arm_compute/runtime/CL/functions/CLLogicalNot.h"
#include "arm_compute/runtime/CL/functions/CLLogicalOr.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/LogicalFixture.h"

namespace
{
using namespace arm_compute;

const auto correct_shape = TensorShape(1, 2, 3, 4); // target shape to check against
const auto wrong_shape   = TensorShape(1, 2, 2, 4); // wrong shape to check validate logic
const auto correct_dt    = DataType::U8;            // correct data type to check against
const auto wrong_dt      = DataType::F32;           // wrong data type to check validate logic
}

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(LogicalOr)
TEST_SUITE(Validate)
TEST_CASE(NullPtr, framework::DatasetMode::ALL)
{
    Status s = CLLogicalOr::validate(nullptr, nullptr, nullptr);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);
}

TEST_CASE(WrongDataType, framework::DatasetMode::ALL)
{
    TensorInfo in1{ correct_shape, 1, correct_dt };
    TensorInfo in2{ correct_shape, 1, wrong_dt };
    TensorInfo out{ correct_shape, 1, correct_dt };

    Status s = CLLogicalOr::validate(&in1, &in2, &out);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Validate
template <typename T>
using CLLogicalOrFixture = LogicalOrValidationFixture<CLTensor, CLAccessor, CLLogicalOr, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLLogicalOrFixture<uint8_t>, framework::DatasetMode::ALL, zip(datasets::SmallShapes(), datasets::SmallShapes()))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLLogicalOrFixture<uint8_t>, framework::DatasetMode::ALL, datasets::SmallShapesBroadcast())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // LogicalOr

TEST_SUITE(LogicalAnd)
TEST_SUITE(Validate)
TEST_CASE(NullPtr, framework::DatasetMode::ALL)
{
    Status s = CLLogicalAnd::validate(nullptr, nullptr, nullptr);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);
}

TEST_CASE(WrongDataType, framework::DatasetMode::ALL)
{
    TensorInfo in1{ correct_shape, 1, correct_dt };
    TensorInfo in2{ correct_shape, 1, wrong_dt };
    TensorInfo out{ correct_shape, 1, correct_dt };

    Status s = CLLogicalAnd::validate(&in1, &in2, &out);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Validate
template <typename T>
using CLLogicalAndFixture = LogicalAndValidationFixture<CLTensor, CLAccessor, CLLogicalAnd, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLLogicalAndFixture<uint8_t>, framework::DatasetMode::ALL, zip(datasets::SmallShapes(), datasets::SmallShapes()))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLLogicalAndFixture<uint8_t>, framework::DatasetMode::ALL, datasets::SmallShapesBroadcast())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // LogicalAnd
TEST_SUITE(LogicalNot)

TEST_SUITE(Validate)
TEST_CASE(NullPtr, framework::DatasetMode::ALL)
{
    Status s = CLLogicalNot::validate(nullptr, nullptr);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);
}

TEST_CASE(WrongDataType, framework::DatasetMode::ALL)
{
    TensorInfo in{ correct_shape, 1, correct_dt };
    TensorInfo out{ correct_shape, 1, wrong_dt };

    Status s = CLLogicalNot::validate(&in, &out);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);

    in  = TensorInfo{ correct_shape, 1, wrong_dt };
    out = TensorInfo{ correct_shape, 1, correct_dt };

    s = CLLogicalNot::validate(&in, &out);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);

    in  = TensorInfo{ correct_shape, 1, wrong_dt };
    out = TensorInfo{ correct_shape, 1, wrong_dt };

    s = CLLogicalNot::validate(&in, &out);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);
}

TEST_CASE(WrongShape, framework::DatasetMode::ALL)
{
    TensorInfo in{ correct_shape, 1, correct_dt };
    TensorInfo out{ wrong_shape, 1, correct_dt };

    Status s = CLLogicalNot::validate(&in, &out);
    ARM_COMPUTE_EXPECT((bool)s == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Validate

template <typename T>
using CLLogicalNotFixture = LogicalNotValidationFixture<CLTensor, CLAccessor, CLLogicalNot, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLLogicalNotFixture<uint8_t>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                    DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // LogicalNot
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
