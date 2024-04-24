/*
 * Copyright (c) 2024 Arm Limited.
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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLScatter.h"
#include "tests/validation/fixtures/ScatterLayerFixture.h"
#include "tests/datasets/ScatterDataset.h"
#include "tests/CL/CLAccessor.h"
#include "arm_compute/function_info/ScatterInfo.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for fp32 data type */
} // namespace

template <typename T>
using CLScatterLayerFixture = ScatterValidationFixture<CLTensor, CLAccessor, CLScatter, T>;

using framework::dataset::make;

TEST_SUITE(CL)
TEST_SUITE(Scatter)
DATA_TEST_CASE(Validate, framework::DatasetMode::PRECOMMIT, zip(
    make("InputInfo", { TensorInfo(TensorShape(9U), 1, DataType::F32),    // Mismatching data types
                        TensorInfo(TensorShape(15U), 1, DataType::F32),   // Valid
                        TensorInfo(TensorShape(8U), 1, DataType::F32),
                        TensorInfo(TensorShape(217U), 1, DataType::F32),    // Mismatch input/output dims.
                        TensorInfo(TensorShape(217U), 1, DataType::F32),    // Updates dim higher than Input/Output dims.
                        TensorInfo(TensorShape(12U), 1, DataType::F32),     // Indices wrong datatype.
                        TensorInfo(TensorShape(9U, 3U, 4U), 1, DataType::F32), // Number of updates != number of indices
                        TensorInfo(TensorShape(17U, 3U, 3U, 2U), 1, DataType::F32), // index_len != (dst_dims - upt_dims + 1)
                        TensorInfo(TensorShape(17U, 3U, 3U, 2U, 2U, 2U), 1, DataType::F32), // index_len > 5
    }),
    make("UpdatesInfo",{TensorInfo(TensorShape(3U), 1, DataType::F16),
                        TensorInfo(TensorShape(15U), 1, DataType::F32),
                        TensorInfo(TensorShape(2U), 1, DataType::F32),
                        TensorInfo(TensorShape(217U), 1, DataType::F32),
                        TensorInfo(TensorShape(217U, 3U), 1, DataType::F32),
                        TensorInfo(TensorShape(2U), 1, DataType::F32),
                        TensorInfo(TensorShape(9U, 3U, 2U), 1, DataType::F32),
                        TensorInfo(TensorShape(17U, 3U, 2U), 1, DataType::F32),
                        TensorInfo(TensorShape(1U), 1, DataType::F32),
    }),
    make("IndicesInfo",{TensorInfo(TensorShape(1U, 3U), 1, DataType::S32),
                        TensorInfo(TensorShape(1U, 15U), 1, DataType::S32),
                        TensorInfo(TensorShape(1U, 2U), 1, DataType::S32),
                        TensorInfo(TensorShape(1U, 271U), 1, DataType::S32),
                        TensorInfo(TensorShape(1U, 271U), 1, DataType::S32),
                        TensorInfo(TensorShape(1U, 2U), 1 , DataType::F32),
                        TensorInfo(TensorShape(1U, 4U), 1, DataType::S32),
                        TensorInfo(TensorShape(3U, 2U), 1, DataType::S32),
                        TensorInfo(TensorShape(6U, 2U), 1, DataType::S32),
    }),
    make("OutputInfo",{TensorInfo(TensorShape(9U), 1, DataType::F16),
                       TensorInfo(TensorShape(15U), 1, DataType::F32),
                       TensorInfo(TensorShape(8U), 1, DataType::F32),
                       TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                       TensorInfo(TensorShape(271U), 1, DataType::F32),
                       TensorInfo(TensorShape(12U), 1, DataType::F32),
                       TensorInfo(TensorShape(9U, 3U, 4U), 1, DataType::F32),
                       TensorInfo(TensorShape(17U, 3U, 3U, 2U), 1, DataType::F32),
                       TensorInfo(TensorShape(17U, 3U, 3U, 2U, 2U, 2U), 1, DataType::F32),
    }),
    make("ScatterInfo",{ ScatterInfo(ScatterFunction::Add, false),
                         ScatterInfo(ScatterFunction::Max, false),
                         ScatterInfo(ScatterFunction::Min, false),
                         ScatterInfo(ScatterFunction::Add, false),
                         ScatterInfo(ScatterFunction::Update, false),
                         ScatterInfo(ScatterFunction::Sub, false),
                         ScatterInfo(ScatterFunction::Sub, false),
                         ScatterInfo(ScatterFunction::Update, false),
                         ScatterInfo(ScatterFunction::Update, false),
    }),
    make("Expected", { false, true, true, false, false, false, false, false, false })),
    input_info, updates_info, indices_info, output_info, scatter_info, expected)
{
    const Status status = CLScatter::validate(&input_info, &updates_info, &indices_info, &output_info, scatter_info);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

const auto allScatterFunctions = make("ScatterFunction",
    {ScatterFunction::Update, ScatterFunction::Add, ScatterFunction::Sub, ScatterFunction::Min, ScatterFunction::Max });

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScatterLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::Small1DScatterDataset(),
        make("DataType", {DataType::F32}),
        allScatterFunctions,
        make("ZeroInit", {false}),
        make("Inplace", {false})))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

// With this test, src should be passed as nullptr.
FIXTURE_DATA_TEST_CASE(RunSmallZeroInit, CLScatterLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::Small1DScatterDataset(),
        make("DataType", {DataType::F32}),
        make("ScatterFunction", {ScatterFunction::Add}),
        make("ZeroInit", {true}),
        make("Inplace", {false})))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

// Updates/src/dst have same no. dims.
FIXTURE_DATA_TEST_CASE(RunSmallMultiDim, CLScatterLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallScatterMultiDimDataset(),
        make("DataType", {DataType::F32}),
        allScatterFunctions,
        make("ZeroInit", {false}),
        make("Inplace", {false})))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

// m+1-D to m+n-D cases
FIXTURE_DATA_TEST_CASE(RunSmallMultiIndices, CLScatterLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallScatterMultiIndicesDataset(),
        make("DataType", {DataType::F32}),
        make("ScatterFunction", {ScatterFunction::Update, ScatterFunction::Add }),
        make("ZeroInit", {false}),
        make("Inplace", {false, true})))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

// m+k, k-1-D m+n-D case
FIXTURE_DATA_TEST_CASE(RunSmallBatchedMultiIndices, CLScatterLayerFixture<float>, framework::DatasetMode::DISABLED,
    combine(datasets::SmallScatterBatchedDataset(),
        make("DataType", {DataType::F32}),
        make("ScatterFunction", {ScatterFunction::Update, ScatterFunction::Add }),
        make("ZeroInit", {false}),
        make("Inplace", {false})))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // Scatter
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
