/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLConcatenateLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConcatenateLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(BatchConcatenateLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo1", {  TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32), // Mismatching data type input/output
                                                  TensorInfo(TensorShape(20U, 27U, 4U, 4U), 1, DataType::F32), // Mismatching x dimension
                                                  TensorInfo(TensorShape(23U, 26U, 4U, 3U), 1, DataType::F32), // Mismatching y dim
                                                  TensorInfo(TensorShape(23U, 27U, 4U, 3U), 1, DataType::F32), // Mismatching z dim
                                                  TensorInfo(TensorShape(16U, 27U, 3U, 6U), 1, DataType::F32)
        }),
        framework::dataset::make("InputInfo2", {  TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32),
                                                  TensorInfo(TensorShape(23U, 27U, 4U, 4U), 1, DataType::F32),
                                                  TensorInfo(TensorShape(23U, 27U, 4U, 4U), 1, DataType::F32),
                                                  TensorInfo(TensorShape(23U, 27U, 3U, 3U), 1, DataType::F32),
                                                  TensorInfo(TensorShape(16U, 27U, 3U, 6U), 1, DataType::F32)
        })),
        framework::dataset::make("OutputInfo", {  TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F16),
                                                  TensorInfo(TensorShape(23U, 12U, 4U, 4U), 1, DataType::F32),
                                                  TensorInfo(TensorShape(23U, 27U, 4U, 4U), 1, DataType::F32),
                                                  TensorInfo(TensorShape(23U, 20U, 4U, 3U), 1, DataType::F32),
                                                  TensorInfo(TensorShape(16U, 27U, 3U, 12U), 1, DataType::F32)
        })),
        framework::dataset::make("Expected", { false, false, false, false, true })),
        input_info1, input_info2, output_info,expected)
{
    std::vector<TensorInfo> inputs_vector_info;
    inputs_vector_info.emplace_back(std::move(input_info1));
    inputs_vector_info.emplace_back(std::move(input_info2));

    std::vector<ITensorInfo *> inputs_vector_info_raw;
    inputs_vector_info_raw.reserve(inputs_vector_info.size());
    for(auto &input : inputs_vector_info)
    {
        inputs_vector_info_raw.emplace_back(&input);
    }

    bool is_valid = bool(CLConcatenateLayer::validate(inputs_vector_info_raw, &output_info.clone()->set_is_resizable(false), 3));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_CASE(Configuration, framework::DatasetMode::ALL)
{
    // Create tensors
    CLTensor src1 = create_tensor<CLTensor>(TensorShape(128U, 32U, 32U), DataType::F32, 1);
    CLTensor src2 = create_tensor<CLTensor>(TensorShape(128U, 32U, 32U), DataType::F32, 1);
    CLTensor src3 = create_tensor<CLTensor>(TensorShape(128U, 32U, 32U), DataType::F32, 1);
    CLTensor dst;

    ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src2.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src3.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLConcatenateLayer       concat_layer;
    std::vector<ICLTensor *> inputs;
    inputs.emplace_back(&src1);
    inputs.emplace_back(&src2);
    inputs.emplace_back(&src3);
    concat_layer.configure(inputs, &dst, 3);
}
template <typename T>
using CLBatchConcatenateLayerFixture = ConcatenateLayerValidationFixture<CLTensor, ICLTensor, CLAccessor, CLConcatenateLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLBatchConcatenateLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(concat(datasets::Small3DShapes(), datasets::Tiny4DShapes()),
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::F16)),
                                                                                                                  framework::dataset::make("Axis", 3)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLBatchConcatenateLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(concat(datasets::Large3DShapes(), datasets::Small4DShapes()),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                framework::dataset::make("Axis", 3)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLBatchConcatenateLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(concat(datasets::Small3DShapes(), datasets::Tiny4DShapes()),
                                                                                                                   framework::dataset::make("DataType",
                                                                                                                           DataType::F32)),
                                                                                                                   framework::dataset::make("Axis", 3)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLBatchConcatenateLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::ConcatenateLayerShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::F32)),
                                                                                                                 framework::dataset::make("Axis", 3)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLBatchConcatenateLayerFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(concat(datasets::Small3DShapes(), datasets::Tiny4DShapes()),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::QASYMM8)),
                                                                                                                     framework::dataset::make("Axis", 3)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLBatchConcatenateLayerFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::ConcatenateLayerShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QASYMM8)),
                                                                                                                   framework::dataset::make("Axis", 3)))
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
