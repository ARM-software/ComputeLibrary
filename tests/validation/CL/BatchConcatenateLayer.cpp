/*
 * Copyright (c) 2019-2020 Arm Limited.
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
namespace
{
/** Zero padding test */
bool validate_zero_padding(unsigned int width, unsigned int height, unsigned int channels, unsigned int batches, DataType data_type)
{
    TensorShape src_shape(width, height, channels, batches);
    TensorShape dst_shape(width, height, channels, batches * 2);

    // Create tensors
    CLTensor src0 = create_tensor<CLTensor>(src_shape, data_type);
    CLTensor src1 = create_tensor<CLTensor>(src_shape, data_type);
    CLTensor dst  = create_tensor<CLTensor>(dst_shape, data_type);

    src0.info()->set_quantization_info(QuantizationInfo(1.f / 256.f, 0));
    src1.info()->set_quantization_info(QuantizationInfo(1.f / 256.f, 0));
    dst.info()->set_quantization_info(QuantizationInfo(1.f / 256.f, 0));

    ARM_COMPUTE_EXPECT(src0.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    std::vector<const ICLTensor *> srcs = { &src0, &src1 };

    // Create and configure function
    CLConcatenateLayer concat;
    concat.configure(srcs, &dst, 3U);

    // Padding can be added along rhs and bias's X dimension
    return src0.info()->padding().empty() && src1.info()->padding().empty() && dst.info()->padding().empty();
}
}
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

    std::vector<const ITensorInfo *> inputs_vector_info_raw;
    inputs_vector_info_raw.reserve(inputs_vector_info.size());
    for(auto &input : inputs_vector_info)
    {
        inputs_vector_info_raw.emplace_back(&input);
    }

    bool is_valid = bool(CLConcatenateLayer::validate(inputs_vector_info_raw, &output_info.clone()->set_is_resizable(false), 3));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

/** Validate zero padding tests
 *
 * A series of validation tests to check that no padding is added as part of configuration for 4 different scenarios.
 *
 * Checks performed in order:
 *     - First dimension multiple of 16
 *     - First dimension non-multiple of 16
 *     - First dimension less than 16 (vec_size for qasymm8) but multiple
 *     - First dimension less than 16 (vec_size for qasymm8) non-multiple
 *     - Tensor with only one element
 */
DATA_TEST_CASE(ValidateZeroPadding, framework::DatasetMode::ALL, zip(
framework::dataset::make("Width",    { 32U, 37U, 12U, 13U, 1U }),
framework::dataset::make("DataType", { DataType::F32, DataType::QASYMM8 })),
width, data_type)
{
    const bool one_elem = (width == 1U);
    bool status = validate_zero_padding(width, one_elem ? 1U : 17U, one_elem ? 1U : 7U, one_elem ? 1U : 2U, data_type);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

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
