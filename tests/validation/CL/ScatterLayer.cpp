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

template <typename T>
using CLScatterLayerFixture = ScatterValidationFixture<CLTensor, CLAccessor, CLScatter, T>;

using framework::dataset::make;

TEST_SUITE(CL)
TEST_SUITE(Scatter)
DATA_TEST_CASE(Validate, framework::DatasetMode::DISABLED, zip(
    make("InputInfo", { TensorInfo(TensorShape(9U), 1, DataType::F32),    // Mismatching data types
                                            TensorInfo(TensorShape(15U), 1, DataType::F32), // Valid
                                            TensorInfo(TensorShape(8U), 1, DataType::F32),
                                            TensorInfo(TensorShape(217U), 1, DataType::F32),    // Mismatch input/output dims.
                                            TensorInfo(TensorShape(217U), 1, DataType::F32),    // Updates dim higher than Input/Output dims.
                                            TensorInfo(TensorShape(12U), 1, DataType::F32),      // Indices wrong datatype.
                                          }),
    make("UpdatesInfo",{                    TensorInfo(TensorShape(3U), 1, DataType::F16),
                                             TensorInfo(TensorShape(15U), 1, DataType::F32),
                                             TensorInfo(TensorShape(2U), 1, DataType::F32),
                                             TensorInfo(TensorShape(217U), 1, DataType::F32),
                                             TensorInfo(TensorShape(217U, 3U), 1, DataType::F32),
                                             TensorInfo(TensorShape(2U), 1, DataType::F32),
                                          }),
    make("IndicesInfo",{                  TensorInfo(TensorShape(3U), 1, DataType::U32),
                                          TensorInfo(TensorShape(15U), 1, DataType::U32),
                                          TensorInfo(TensorShape(2U), 1, DataType::U32),
                                          TensorInfo(TensorShape(271U), 1, DataType::U32),
                                          TensorInfo(TensorShape(271U), 1, DataType::U32),
                                          TensorInfo(TensorShape(2U), 1 , DataType::S32)
                                          }),
    make("OutputInfo",{                     TensorInfo(TensorShape(9U), 1, DataType::F16),
                                            TensorInfo(TensorShape(15U), 1, DataType::F32),
                                            TensorInfo(TensorShape(8U), 1, DataType::F32),
                                            TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(271U), 1, DataType::F32),
                                            TensorInfo(TensorShape(12U), 1, DataType::F32)
                                           }),
    make("ScatterInfo",{ ScatterInfo(ScatterFunction::Add, false),
                                           }),
    make("Expected", { false, true, true, false, false, false })),
    input_info, updates_info, indices_info, output_info, scatter_info, expected)
{
    // TODO: Enable validation tests.
    ARM_COMPUTE_UNUSED(input_info);
    ARM_COMPUTE_UNUSED(updates_info);
    ARM_COMPUTE_UNUSED(indices_info);
    ARM_COMPUTE_UNUSED(output_info);
    ARM_COMPUTE_UNUSED(scatter_info);
    ARM_COMPUTE_UNUSED(expected);
}

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScatterLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small1DScatterDataset(),
                                                                                                                    make("DataType", {DataType::F32}),
                                                                                                                    make("ScatterFunction", {ScatterFunction::Update, ScatterFunction::Add, ScatterFunction::Sub, ScatterFunction::Min, ScatterFunction::Max}),
                                                                                                                    make("ZeroInit", {false})))
{
    // TODO: Add validate() here.
}

// With this test, src should be passed as nullptr.
FIXTURE_DATA_TEST_CASE(RunSmallZeroInit, CLScatterLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small1DScatterDataset(),
                                                                                                                    make("DataType", {DataType::F32}),
                                                                                                                    make("ScatterFunction", {ScatterFunction::Add}),
                                                                                                                    make("ZeroInit", {true})))
{
    // TODO: Add validate() here
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // Scatter
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
