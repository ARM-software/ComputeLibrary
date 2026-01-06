/*
 * Copyright (c) 2024-2026 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/functions/NEScatter.h"

#include "tests/datasets/DatatypeDataset.h"
#include "tests/datasets/ScatterDataset.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/ScatterLayerFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename T>
using NEScatterLayerFixture = ScatterValidationFixture<Tensor, Accessor, NEScatter, T>;
namespace
{
RelativeTolerance<float> tolerance_f32(
    0.001f); /**< Tolerance value for comparing reference's output against implementation's output for fp32 data type */
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<float> tolerance_f16(
    0.02f); /**< Tolerance value for comparing reference's output against implementation's output for fp16 data type */
#endif      // ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<int32_t> tolerance_int(
    0); /**< Tolerance value for comparing reference's output against implementation's output for integer data types */
} // namespace
using framework::dataset::make;

void validate_data_types(DataType input_dtype, DataType updates_dtype, DataType indices_dtype, DataType output_dtype)
{
    const auto input   = TensorInfo(TensorShape(6U, 5U, 2U), 1, input_dtype);
    const auto updates = TensorInfo(TensorShape(6U, 4U), 1, updates_dtype);
    const auto indices = TensorInfo(TensorShape(2U, 4U), 1, indices_dtype);
    auto       output  = TensorInfo(TensorShape(6U, 5U, 2U), 1, output_dtype);

    ScatterInfo scatter_info = ScatterInfo(ScatterFunction::Update, false);

    const bool is_valid = static_cast<bool>(NEScatter::validate(&input, &updates, &indices, &output, scatter_info));

    const auto supports = {std::make_tuple(DataType::F32, DataType::F32, DataType::S32, DataType::F32),
                           std::make_tuple(DataType::F16, DataType::F16, DataType::S32, DataType::F16),
                           std::make_tuple(DataType::S32, DataType::S32, DataType::S32, DataType::S32),
                           std::make_tuple(DataType::S16, DataType::S16, DataType::S32, DataType::S16),
                           std::make_tuple(DataType::S8, DataType::S8, DataType::S32, DataType::S8),
                           std::make_tuple(DataType::U32, DataType::U32, DataType::S32, DataType::U32),
                           std::make_tuple(DataType::U16, DataType::U16, DataType::S32, DataType::U16),
                           std::make_tuple(DataType::U8, DataType::U8, DataType::S32, DataType::U8)};
    const auto config   = std::make_tuple(input_dtype, updates_dtype, indices_dtype, output_dtype);
    const std::initializer_list<DataType> dtypes_list = {input_dtype, updates_dtype, indices_dtype, output_dtype};

    bool expected = false;
    if (cpu_supports_dtypes(dtypes_list))
    {
        expected = (std::find(supports.begin(), supports.end(), config) != supports.end());
    }
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE(NEON)
TEST_SUITE(Scatter)
DATA_TEST_CASE(
    Validate,
    framework::DatasetMode::PRECOMMIT,
    zip(make("InputInfo",
             {
                 TensorInfo(TensorShape(9U), 1, DataType::F32),  // Mismatching data types
                 TensorInfo(TensorShape(15U), 1, DataType::F32), // Valid
                 TensorInfo(TensorShape(15U), 1, DataType::U8),  // Valid
                 TensorInfo(TensorShape(8U), 1, DataType::F32),
                 TensorInfo(TensorShape(217U), 1, DataType::F32),       // Mismatch input/output dims.
                 TensorInfo(TensorShape(217U), 1, DataType::F32),       // Updates dim higher than Input/Output dims.
                 TensorInfo(TensorShape(12U), 1, DataType::F32),        // Indices wrong datatype.
                 TensorInfo(TensorShape(9U, 3U, 4U), 1, DataType::F32), // Number of updates != number of indices
                 TensorInfo(TensorShape(17U, 3U, 3U, 2U), 1, DataType::F32), // index_len != (dst_dims - upt_dims + 1)
                 TensorInfo(TensorShape(17U, 3U, 3U, 2U, 2U, 2U), 1, DataType::F32), // index_len > 5
             }),
        make("UpdatesInfo",
             {
                 TensorInfo(TensorShape(3U), 1, DataType::F16),
                 TensorInfo(TensorShape(15U), 1, DataType::F32),
                 TensorInfo(TensorShape(15U), 1, DataType::U8),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(217U), 1, DataType::F32),
                 TensorInfo(TensorShape(217U, 3U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(9U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(17U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(1U), 1, DataType::F32),
             }),
        make("IndicesInfo",
             {
                 TensorInfo(TensorShape(1U, 3U), 1, DataType::S32),
                 TensorInfo(TensorShape(1U, 15U), 1, DataType::S32),
                 TensorInfo(TensorShape(1U, 15U), 1, DataType::S32),
                 TensorInfo(TensorShape(1U, 2U), 1, DataType::S32),
                 TensorInfo(TensorShape(1U, 271U), 1, DataType::S32),
                 TensorInfo(TensorShape(1U, 271U), 1, DataType::S32),
                 TensorInfo(TensorShape(1U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(1U, 4U), 1, DataType::S32),
                 TensorInfo(TensorShape(3U, 2U), 1, DataType::S32),
                 TensorInfo(TensorShape(6U, 2U), 1, DataType::S32),
             }),
        make("OutputInfo",
             {
                 TensorInfo(TensorShape(9U), 1, DataType::F16),
                 TensorInfo(TensorShape(15U), 1, DataType::F32),
                 TensorInfo(TensorShape(15U), 1, DataType::U8),
                 TensorInfo(TensorShape(8U), 1, DataType::F32),
                 TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                 TensorInfo(TensorShape(271U), 1, DataType::F32),
                 TensorInfo(TensorShape(12U), 1, DataType::F32),
                 TensorInfo(TensorShape(9U, 3U, 4U), 1, DataType::F32),
                 TensorInfo(TensorShape(17U, 3U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(17U, 3U, 3U, 2U, 2U, 2U), 1, DataType::F32),
             }),
        make("ScatterInfo",
             {
                 ScatterInfo(ScatterFunction::Add, false),
                 ScatterInfo(ScatterFunction::Max, false),
                 ScatterInfo(ScatterFunction::Max, false),
                 ScatterInfo(ScatterFunction::Min, false),
                 ScatterInfo(ScatterFunction::Add, false),
                 ScatterInfo(ScatterFunction::Update, false),
                 ScatterInfo(ScatterFunction::Sub, false),
                 ScatterInfo(ScatterFunction::Sub, false),
                 ScatterInfo(ScatterFunction::Update, false),
                 ScatterInfo(ScatterFunction::Update, false),
             }),
        make("Expected", {false, true, true, true, false, false, false, false, false, false})),
    input_info,
    updates_info,
    indices_info,
    output_info,
    scatter_info,
    expected)
{
    const Status status = NEScatter::validate(&input_info, &updates_info, &indices_info, &output_info, scatter_info);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(ValidateAllDataTypes,
               framework::DatasetMode::NIGHTLY,
               combine(datasets::AllDataTypes("InputDataType"),
                       datasets::AllDataTypes("UpdatesDataType"),
                       datasets::AllDataTypes("IndicesDataType"),
                       datasets::AllDataTypes("OutputDataType")),
               input_dtype,
               updates_dtype,
               indices_dtype,
               output_dtype)
{
    validate_data_types(input_dtype, updates_dtype, indices_dtype, output_dtype);
}

DATA_TEST_CASE(ValidateCommonDataTypes,
               framework::DatasetMode::PRECOMMIT,
               combine(datasets::CommonDataTypes("InputDataType"),
                       datasets::CommonDataTypes("UpdatesDataType"),
                       datasets::CommonDataTypes("IndicesDataType"),
                       datasets::CommonDataTypes("OutputDataType")),
               input_dtype,
               updates_dtype,
               indices_dtype,
               output_dtype)
{
    validate_data_types(input_dtype, updates_dtype, indices_dtype, output_dtype);
}

const auto allScatterFunctions = make(
    "ScatterFunction",
    {ScatterFunction::Update, ScatterFunction::Add, ScatterFunction::Sub, ScatterFunction::Min, ScatterFunction::Max});

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEScatterLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small1DScatterDataset(),
                               make("DataType", {DataType::F32}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false, true})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

// With this test, src should be passed as nullptr.
FIXTURE_DATA_TEST_CASE(RunSmallZeroInit,
                       NEScatterLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::Small1DScatterDataset(),
                               make("DataType", {DataType::F32}),
                               make("ScatterFunction", {ScatterFunction::Add}),
                               make("ZeroInit", {true}),
                               make("Inplace", {false}),
                               make("Padding", {true})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

// Updates/src/dst have same no. dims.
FIXTURE_DATA_TEST_CASE(RunSmallMultiDim,
                       NEScatterLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMultiDimDataset(),
                               make("DataType", {DataType::F32}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false, true})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

// m+1-D to m+n-D cases
FIXTURE_DATA_TEST_CASE(RunSmallMultiIndices,
                       NEScatterLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMultiIndicesDataset(),
                               make("DataType", {DataType::F32}),
                               make("ScatterFunction", {ScatterFunction::Update, ScatterFunction::Add}),
                               make("ZeroInit", {false}),
                               make("Inplace", {false, true}),
                               make("Padding", {false, true})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

// m+k, k-1-D m+n-D case
FIXTURE_DATA_TEST_CASE(RunSmallBatchedMultiIndices,
                       NEScatterLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterBatchedDataset(),
                               make("DataType", {DataType::F32}),
                               make("ScatterFunction", {ScatterFunction::Update, ScatterFunction::Add}),
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false, true})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

// m+k, k-1-D m+n-D case
FIXTURE_DATA_TEST_CASE(RunSmallScatterScalar,
                       NEScatterLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterScalarDataset(),
                               make("DataType", {DataType::F32}),
                               make("ScatterFunction", {ScatterFunction::Update, ScatterFunction::Add}),
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false}))) // NOTE: Padding not supported in this datset
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // FP32

// NOTE: Padding is disabled for the SmallScatterMixedDataset due certain shapes not supporting padding.
//       Padding is well tested in F32 Datatype test cases.
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmallMixed,
                       NEScatterLayerFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMixedDataset(),
                               make("DataType", {DataType::F16}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false})))
{
    if (CPUInfo::get().has_fp16())
    {
        validate(Accessor(_target), _reference, tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           // ARM_COMPUTE_ENABLE_FP16
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmallMixed,
                       NEScatterLayerFixture<int32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMixedDataset(),
                               make("DataType", {DataType::S32}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false})))
{
    validate(Accessor(_target), _reference, tolerance_int);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmallMixed,
                       NEScatterLayerFixture<int16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMixedDataset(),
                               make("DataType", {DataType::S16}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false})))
{
    validate(Accessor(_target), _reference, tolerance_int);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmallMixed,
                       NEScatterLayerFixture<int8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMixedDataset(),
                               make("DataType", {DataType::S8}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false})))
{
    validate(Accessor(_target), _reference, tolerance_int);
}
TEST_SUITE_END() // S8

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmallMixed,
                       NEScatterLayerFixture<uint32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMixedDataset(),
                               make("DataType", {DataType::U32}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false})))
{
    validate(Accessor(_target), _reference, tolerance_int);
}
TEST_SUITE_END() // U32

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmallMixed,
                       NEScatterLayerFixture<uint16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMixedDataset(),
                               make("DataType", {DataType::U16}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false})))
{
    validate(Accessor(_target), _reference, tolerance_int);
}
TEST_SUITE_END() // U16

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmallMixed,
                       NEScatterLayerFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallScatterMixedDataset(),
                               make("DataType", {DataType::U8}),
                               allScatterFunctions,
                               make("ZeroInit", {false}),
                               make("Inplace", {false}),
                               make("Padding", {false})))
{
    validate(Accessor(_target), _reference, tolerance_int);
}
TEST_SUITE_END() // U8
TEST_SUITE_END() // Integer

TEST_SUITE_END() // Scatter
TEST_SUITE_END() // NEON

} // namespace validation
} // namespace test
} // namespace arm_compute
