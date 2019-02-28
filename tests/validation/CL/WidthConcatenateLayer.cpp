/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLWidthConcatenateLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WidthConcatenateLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(WidthConcatenateLayer)
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
              framework::dataset::make("InputInfo1", {  TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching data type input/output
                                                        TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching y dimension
                                                        TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching total width
                                                        TensorInfo(TensorShape(16U, 27U, 5U), 1, DataType::F32)
              }),
              framework::dataset::make("InputInfo2", {  TensorInfo(TensorShape(24U, 27U, 4U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(52U, 27U, 5U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(52U, 27U, 5U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 27U, 5U), 1, DataType::F32)
              })),
              framework::dataset::make("OutputInfo", {  TensorInfo(TensorShape(47U, 27U, 5U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(75U, 12U, 5U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U, 27U, 5U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 27U, 5U), 1, DataType::F32)
              })),
              framework::dataset::make("Expected", { false, false, false, true })),
              input_info1, input_info2, output_info,expected)
{
    std::vector<TensorInfo> inputs_vector_info;
    inputs_vector_info.emplace_back(std::move(input_info1));
    inputs_vector_info.emplace_back(std::move(input_info2));

    std::vector<ITensorInfo *> inputs_vector_info_raw;
    for(auto &input : inputs_vector_info)
    {
        inputs_vector_info_raw.emplace_back(&input);
    }

    bool is_valid = bool(CLWidthConcatenateLayer::validate(inputs_vector_info_raw,
                                                           &output_info.clone()->set_is_resizable(false)));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_CASE(Configuration, framework::DatasetMode::ALL)
{
    // Create tensors
    CLTensor src1 = create_tensor<CLTensor>(TensorShape(128U, 32U, 32U), DataType::F32, 1);
    CLTensor src2 = create_tensor<CLTensor>(TensorShape(32U, 32U, 32U), DataType::F32, 1);
    CLTensor src3 = create_tensor<CLTensor>(TensorShape(15U, 32U, 32U), DataType::F32, 1);
    CLTensor dst;

    ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src2.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src3.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWidthConcatenateLayer concat_layer;

    concat_layer.configure({ &src1, &src2, &src3 }, &dst);
}

template <typename T>
using CLWidthConcatenateLayerFixture = WidthConcatenateLayerValidationFixture<CLTensor, ICLTensor, CLAccessor, CLWidthConcatenateLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWidthConcatenateLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small2DShapes(), datasets::Tiny4DShapes()),
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWidthConcatenateLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large2DShapes(), datasets::Small4DShapes()),
                                                                                                                framework::dataset::make("DataType",
                                                                                                                        DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWidthConcatenateLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small2DShapes(), datasets::Tiny4DShapes()),
                                                                                                                   framework::dataset::make("DataType",
                                                                                                                           DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWidthConcatenateLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::WidthConcatenateLayerShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWidthConcatenateLayerFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small2DShapes(), datasets::Tiny4DShapes()),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::QASYMM8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWidthConcatenateLayerFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::WidthConcatenateLayerShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QASYMM8)))
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
