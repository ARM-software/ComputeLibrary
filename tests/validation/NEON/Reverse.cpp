/*
 * Copyright (c) 2018-2021, 2023-2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEReverse.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/DatatypeDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReverseFixture.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
using framework::dataset::make;

auto run_small_dataset = combine(datasets::Small3DShapes(), datasets::Tiny1DShapes());
auto run_large_dataset = combine(datasets::LargeShapes(), datasets::Tiny1DShapes());

void validate_data_types(DataType input_dtype, DataType output_dtype, DataType axis_dtype)
{
    const auto input = TensorInfo(TensorShape(16U, 16U, 5U), 1, input_dtype);
    const auto axis = TensorInfo(TensorShape(1U), 1, axis_dtype);
    auto output = TensorInfo(TensorShape(16U, 16U, 5U), 1, output_dtype);

    const Status status = (NEReverse::validate(&input, &output, &axis, false /* use_inverted_axis */));
    const bool is_valid = static_cast<bool>(status);

    static const auto supported_dtypes = {
        DataType::QSYMM8,
        DataType::QASYMM8,
        DataType::QASYMM8_SIGNED,
        DataType::QSYMM16,
        DataType::U8,
        DataType::S8,
        DataType::QSYMM8_PER_CHANNEL,
        DataType::U16,
        DataType::S16,
        DataType::QSYMM16,
        DataType::QASYMM16,
        DataType::U32,
        DataType::S32,
        DataType::BFLOAT16,
        DataType::F16,
        DataType::F32
    };

    static std::vector<std::tuple<DataType,DataType,DataType>> supports = {};
    for(DataType dtype : supported_dtypes)
    {
        supports.push_back(std::make_tuple(dtype, dtype, DataType::S32));
        supports.push_back(std::make_tuple(dtype, dtype, DataType::U32));
    }

    const auto config = std::make_tuple(input_dtype, output_dtype, axis_dtype);
    const bool expected = (std::find(supports.begin(), supports.end(), config) != supports.end());

    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Reverse)

/// @note: Do not modify. Validating all data types is pretty fast.
DATA_TEST_CASE(ValidateAllDataTypes, framework::DatasetMode::ALL,
    combine(
        datasets::AllDataTypes("InputDataType"),
        datasets::AllDataTypes("OutputDataType"),
        datasets::AllDataTypes("AxisDataType")),
        input_dtype, output_dtype, axis_dtype)
{
    validate_data_types(input_dtype, output_dtype, axis_dtype);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
        make("InputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8), // Invalid axis datatype
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Invalid axis shape
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Invalid axis length (> 4)
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Mismatching shapes
                                            TensorInfo(TensorShape(32U, 13U, 17U, 3U, 2U), 1, DataType::U8), // Unsupported source dimensions (>4)
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U), 1, DataType::U8),
        }),
        make("OutputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(32U, 13U, 17U, 3U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U), 1, DataType::U8),
        }),
        make("AxisInfo", { TensorInfo(TensorShape(3U), 1, DataType::U8),
                                           TensorInfo(TensorShape(2U, 10U), 1, DataType::U32),
                                           TensorInfo(TensorShape(8U), 1, DataType::U32),
                                           TensorInfo(TensorShape(2U), 1, DataType::U32),
                                           TensorInfo(TensorShape(2U), 1, DataType::U32),
                                           TensorInfo(TensorShape(2U), 1, DataType::U32),
                                           TensorInfo(TensorShape(2U), 1, DataType::U32),
        }),
        make("Expected", { false, false, false, false, false, true, true})
        ),
        src_info, dst_info, axis_info, expected)
{
    Status s = NEReverse::validate(&src_info.clone()->set_is_resizable(false),
                                  &dst_info.clone()->set_is_resizable(false),
                                  &axis_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEReverseFixture = ReverseValidationFixture<Tensor, Accessor, NEReverse, T>;

/// @note: Test Strategy --
///    The operator uses uint8_t, uint16_t and uint32_t under the hood depending
///    on the size of the input data type. Therefore, we do not extensively test
///    all the data types here. fp32/16 and qasymm8 has been thoroughly tested with
///    multiple shapes and configuration. Other data types are just smoke tested
///    with a very limited set of configurations, just to make sure they function
///    correctly.

TEST_SUITE(Float)
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           run_small_dataset,
                           make("DataType", {DataType::F16, DataType::BFLOAT16}),
                           make("use_negative_axis", { true, false }),
                           make("use_inverted_axis", { true, false })))
{
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReverseFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(
                           run_large_dataset,
                           make("DataType", DataType::F16),
                           make("use_negative_axis", { true, false }),
                           make("use_inverted_axis", { true, false })))
{
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F16

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           run_small_dataset,
                           make("DataType", DataType::F32),
                           make("use_negative_axis", { true, false }),
                           make("use_inverted_axis", { true, false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReverseFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(
                           run_large_dataset,
                           make("DataType", DataType::F32),
                           make("use_negative_axis", { true, false }),
                           make("use_inverted_axis", { true, false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(Int32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<int32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           make("InOutShape", TensorShape(18U, 5U, 5U)),
                           make("AxisShape", TensorShape(2U)),
                           make("DataType", {DataType::S32}),
                           make("use_negative_axis", { false }),
                           make("use_inverted_axis", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // Int32

TEST_SUITE(UInt32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<uint32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           make("InOutShape", TensorShape(18U, 5U, 5U)),
                           make("AxisShape", TensorShape(2U)),
                           make("DataType", {DataType::U32}),
                           make("use_negative_axis", { false }),
                           make("use_inverted_axis", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // UInt32

TEST_SUITE(Int16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<int16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           make("InOutShape", TensorShape(18U, 5U, 5U)),
                           make("AxisShape", TensorShape(2U)),
                           make("DataType", {DataType::S16, DataType::QSYMM16}),
                           make("use_negative_axis", { false }),
                           make("use_inverted_axis", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // Int16

TEST_SUITE(UInt16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<uint16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           make("InOutShape", TensorShape(18U, 5U, 5U)),
                           make("AxisShape", TensorShape(2U)),
                           make("DataType", {DataType::U16, DataType::QASYMM16}),
                           make("use_negative_axis", { false }),
                           make("use_inverted_axis", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // UInt16

TEST_SUITE(UInt8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           run_small_dataset,
                           make("DataType", {DataType::QASYMM8, DataType::U8}),
                           make("use_negative_axis", { true, false }),
                           make("use_inverted_axis", { true, false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReverseFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(
                           run_large_dataset,
                           make("DataType", DataType::QASYMM8),
                           make("use_negative_axis", { true, false }),
                           make("use_inverted_axis", { true, false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // UInt8

TEST_SUITE(Int8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<int8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                           make("InOutShape", TensorShape(18U, 5U, 5U)),
                           make("AxisShape", TensorShape(2U)),
                           make("DataType", {DataType::QASYMM8_SIGNED, DataType::S8,
                                DataType::QSYMM8, DataType::QSYMM8_PER_CHANNEL}),
                           make("use_negative_axis", { false }),
                           make("use_inverted_axis", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // Int8
TEST_SUITE_END() // Integer

TEST_SUITE_END() // Reverse
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
