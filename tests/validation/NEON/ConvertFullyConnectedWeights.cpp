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
#include "arm_compute/runtime/NEON/functions/NEConvertFullyConnectedWeights.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvertFullyConnectedWeightsFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
auto params = combine(framework::dataset::make("WeightsWidth", { 16, 32, 64 }), framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ConvertFullyConnectedWeights)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 42U), 1, DataType::F32),     // Mismatching data types
                                            TensorInfo(TensorShape(32U, 42U), 1, DataType::F32),     // Valid
                                            TensorInfo(TensorShape(27U, 42U), 1, DataType::F32),     // Mismatching shapes
                                            TensorInfo(TensorShape(27U, 42U), 1, DataType::F32),     // Wrong DataLayout
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 42U), 1, DataType::F16),
                                            TensorInfo(TensorShape(32U, 42U), 1, DataType::F32),
                                            TensorInfo(TensorShape(32U, 42U), 1, DataType::F32),
                                            TensorInfo(TensorShape(32U, 42U), 1, DataType::F32),
                                          })),
    framework::dataset::make("OriginalInput", { TensorShape(7U, 3U, 2U),
                                                TensorShape(7U, 3U, 2U),
                                                TensorShape(7U, 3U, 2U),
                                                TensorShape(7U, 3U, 2U),
                                               })),
    framework::dataset::make("DataLayout", { DataLayout::NCHW,
                                             DataLayout::NCHW,
                                             DataLayout::NCHW,
                                             DataLayout::UNKNOWN,
                                               })),
    framework::dataset::make("Expected", { false, true, false, false})),
    input_info, output_info, original_input_shape, data_layout, expected)
{
    bool is_valid = bool(NEConvertFullyConnectedWeights::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), original_input_shape, data_layout));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEConvertFullyConnectedWeightsFixture = ConvertFullyConnectedWeightsValidationFixture<Tensor, Accessor, NEConvertFullyConnectedWeights, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvertFullyConnectedWeightsFixture<float>, framework::DatasetMode::ALL, combine(datasets::Small3DShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                    DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvertFullyConnectedWeightsFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::Large3DShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                        DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvertFullyConnectedWeightsFixture<half>, framework::DatasetMode::ALL, combine(datasets::Small3DShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                   DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvertFullyConnectedWeightsFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::Large3DShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                       DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvertFullyConnectedWeightsFixture<uint8_t>, framework::DatasetMode::ALL, combine(datasets::Small3DShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                      DataType::QASYMM8))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvertFullyConnectedWeightsFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large3DShapes(), combine(params,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE_END() // ConvertFullyConnectedWeights
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
