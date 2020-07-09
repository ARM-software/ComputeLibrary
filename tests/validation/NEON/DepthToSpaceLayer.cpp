/*
 * Copyright (c) 2019 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDepthToSpaceLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DepthToSpaceDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DepthToSpaceLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(DepthToSpaceLayer)

template <typename T>
using NEDepthToSpaceLayerFixture = DepthToSpaceLayerValidationFixture<Tensor, Accessor, NEDepthToSpaceLayer, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(16U, 8U, 4U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 8U, 4U, 4U), 1, DataType::F32),   // block < 2
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 4U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(16U, 8U, 2U, 4U), 1, DataType::F32),    // Negative block shape
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 4U, 4U), 1, DataType::F32), // Wrong tensor shape
                                                     }),
               framework::dataset::make("BlockShape", { 2, 1, 2, 2, 2 })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 16U, 1U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 16U, 1U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 16U, 2U, 1U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 8U, 2U, 1U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, false, false, false, false})),
               input_info, block_shape, output_info, expected)
{
    bool has_error = bool(NEDepthToSpaceLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), block_shape));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthToSpaceLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallDepthToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                       DataType::F32)),
                                                                                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthToSpaceLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeDepthToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                     DataType::F32)),
                                                                                                             framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthToSpaceLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallDepthToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                      DataType::F16)),
                                                                                                              framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthToSpaceLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeDepthToSpaceLayerDataset(), framework::dataset::make("DataType",
                                                                                                                    DataType::F16)),
                                                                                                            framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END() // DepthToSpace
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
