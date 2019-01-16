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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DilatedConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const AbsoluteTolerance<float>            abs_tolerance_f16(0.3f);                   /**< Absolute tolerance value for comparing reference's output against implementation's output for DataType::F16 */
const RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2f)); /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                           tolerance_num_f16 = 0.07f;                 /**< Tolerance number for FP16 */
#endif                                                                               /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
constexpr AbsoluteTolerance<float> tolerance_qasymm8(0.0);                           /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
    DataType::QASYMM8,
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DilatedConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
                                          framework::dataset::make("InputInfo", { TensorInfo(TensorShape(8U, 8U, 2U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(3U, 3U, 2U, 1U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32)
                                          }),
                                          framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16)
                                          })),
                                          framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(6U, 6U, 1U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32)
                                          })),
                                          framework::dataset::make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(2, 1, 0, 0),
                                                                                 PadStrideInfo(3, 2, 1, 0)
                                          })),
                                          framework::dataset::make("Dilation", { Size2D(1U, 2U),
                                                                                 Size2D(2U, 1U),
                                                                                 Size2D(2U, 2U),
                                                                                 Size2D(3U, 3U)
                                          })),
                                          framework::dataset::make("Expected", { ConvolutionMethod::GEMM, ConvolutionMethod::GEMM, ConvolutionMethod::GEMM, ConvolutionMethod::GEMM })),
               input_info, weights_info, output_info, conv_info, dilation, expected)
{
    ConvolutionMethod is_valid = NEConvolutionLayer::get_convolution_method(&input_info.clone()->set_is_resizable(false),
                                                                            &weights_info.clone()->set_is_resizable(false),
                                                                            &output_info.clone()->set_is_resizable(false),
                                                                            conv_info, WeightsInfo(), dilation);
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // DilatedConvolutionLayer

TEST_SUITE(GEMMDilatedConvolutionLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallDilatedConvolutionLayerDataset(),
                                                                   CNNDataTypes),
               input_shape, weights_shape, bias_shape, output_shape, info, dilation, data_type)
{
    auto bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

    // Create tensors
    Tensor src     = create_tensor<Tensor>(input_shape, data_type, 1, QuantizationInfo(2.f / 255.f, 127));
    Tensor weights = create_tensor<Tensor>(weights_shape, data_type, 1, QuantizationInfo(2.f / 255.f, 127));
    Tensor bias    = create_tensor<Tensor>(bias_shape, bias_data_type, 1, QuantizationInfo(2.f / 255.f, 127));
    Tensor dst     = create_tensor<Tensor>(output_shape, data_type, 1, QuantizationInfo(2.f / 255.f, 127));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    const QuantizationInfo src_quantization_info     = src.info()->quantization_info();
    const QuantizationInfo weights_quantization_info = weights.info()->quantization_info();

    // Create and configure function
    NEGEMMConvolutionLayer conv;
    conv.configure(&src, &weights, &bias, &dst, info, WeightsInfo(), dilation);

    // Validate valid region
    const ValidRegion src_valid_region     = shape_to_valid_region(input_shape);
    const ValidRegion weights_valid_region = shape_to_valid_region(weights_shape);
    const ValidRegion bias_valid_region    = shape_to_valid_region(bias_shape);
    const ValidRegion dst_valid_region     = shape_to_valid_region(output_shape);

    validate(src.info()->valid_region(), src_valid_region);
    validate(weights.info()->valid_region(), weights_valid_region);
    validate(bias.info()->valid_region(), bias_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate QuantizationInfo
    ARM_COMPUTE_EXPECT(src.info()->quantization_info() == src_quantization_info, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->quantization_info() == weights_quantization_info, framework::LogLevel::ERRORS);

    // Validate padding
    //TODO(COMPMID-415) Need to validate padding?
}

template <typename T>
using NEGEMMDilatedConvolutionLayerFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolutionLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMDilatedConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallDilatedConvolutionLayerDataset(),
                                                                                                                        framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                        framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMDilatedConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDilatedConvolutionLayerDataset(),
                                                                                                                      framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                      framework::dataset::make("DataType", DataType::F16)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                      framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMDilatedConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallDilatedConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                       framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMDilatedConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDilatedConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::F32)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                       framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using NEGEMMDilatedConvolutionLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<Tensor, Accessor, NEGEMMConvolutionLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMDilatedConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(datasets::SmallDilatedConvolutionLayerDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                               framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMDilatedConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(datasets::LargeDilatedConvolutionLayerDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                               framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // GEMMDilatedConvolutionLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
