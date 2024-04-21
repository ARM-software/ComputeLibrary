/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPriorBoxLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/PriorBoxLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PriorBoxLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f32(0.00001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(PriorBoxLayer)

template <typename T>
using NEPriorBoxLayerFixture = PriorBoxLayerValidationFixture<Tensor, Accessor, NEPriorBoxLayer, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32)
                                                     }),
               framework::dataset::make("Input2Info", { TensorInfo(TensorShape(10U, 10U, 2U), 1, DataType::F32)
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(1200U, 2U), 1, DataType::F32)
                                                     })),
               framework::dataset::make("PriorBoxInfo",{ PriorBoxLayerInfo(std::vector<float>(1), std::vector<float>(1), 0, true, true, std::vector<float>(1), std::vector<float>(1), Coordinates2D{8, 8}, std::array<float, 2>())
                                                     })),
               framework::dataset::make("Expected", { true})),
               input1_info, input2_info, output_info, info, expected)
{
    bool has_error = bool(NEPriorBoxLayer::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), info));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPriorBoxLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallPriorBoxLayerDataset(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                           framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32, 0);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPriorBoxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargePriorBoxLayerDataset(),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)),
                                                                                                         framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32, 0);
}
TEST_SUITE_END() // Float
TEST_SUITE_END() // FP32

TEST_SUITE_END() // PriorBoxLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
