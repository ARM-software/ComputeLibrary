/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeConvolutionLayerDataset.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/datasets/TinyConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"
#include "tests/validation/fixtures/WinogradConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float>       rel_tolerance_f32(0.01f);  /**< Relative tolerance for FP32 types */
const AbsoluteTolerance<float> abs_tolerance_f32(0.002f); /**< Absolute tolerance for FP32 types */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const AbsoluteTolerance<float> tolerance_f16(0.01f);       /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif                                                     /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const AbsoluteTolerance<float>     tolerance_q(1.0f);      /**< Tolerance value for comparing reference's output against implementation's output for fixed point data types */
constexpr AbsoluteTolerance<float> tolerance_qasymm8(0.0); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
    DataType::QS8,
    DataType::QS16,
    DataType::QASYMM8,
});
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f)
});
} // namespace

TEST_SUITE(NEON)

TEST_SUITE(ConvolutionLayer)
DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
                                                                                           framework::dataset::make("InputInfo", { TensorInfo(TensorShape(18U, 18U, 32U), 1, DataType::F32, 0),
                                                                                                                    TensorInfo(TensorShape(23U, 27U, 32U, 4U), 1, DataType::F32, 0),
                                                                                                                    TensorInfo(TensorShape(3U, 3U, 2U, 1U), 1, DataType::F32, 0),
                                                                                                                    TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32, 0)
                                                                                                                                 }),
                                                                                           framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 32U, 21U), 1, DataType::F32, 0),
                                                                                                                    TensorInfo(TensorShape(5U, 5U, 32U, 21U), 1, DataType::F32, 0),
                                                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32, 0),
                                                                                                                    TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16, 0)
                                                                                                                                   })),
                                                                                       framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(16U, 16U, 21U), 1, DataType::F32, 0),
                                                                                                                TensorInfo(TensorShape(19U, 23U, 21U, 4U), 1, DataType::F32, 0),
                                                                                                                TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32, 0),
                                                                                                                TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32, 0)
                                                                                                                              })),
                                                                                   framework::dataset::make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
                                                                                                            PadStrideInfo(1, 1, 0, 0),
                                                                                                            PadStrideInfo(2, 1, 0, 0),
                                                                                                            PadStrideInfo(3, 2, 1, 0)
                                                                                                                        })),
                                                                               framework::dataset::make("FastMath", { true,
                                                                                                                      true,
                                                                                                                      false,
                                                                                                                      false
                                                                                                                    })),
                                                                           framework::dataset::make("Expected", { ConvolutionMethod::WINOGRAD, ConvolutionMethod::WINOGRAD, ConvolutionMethod::GEMM, ConvolutionMethod::GEMM })),
               input_info, weights_info, output_info, conv_info, fast_math, expected)
{
    ConvolutionMethod is_valid = NEConvolutionLayer::get_convolution_method(&input_info.clone()->set_is_resizable(true),
                                                                            &weights_info.clone()->set_is_resizable(true),
                                                                            &output_info.clone()->set_is_resizable(true), conv_info, WeightsInfo(), Size2D(1U, 1U), ActivationLayerInfo(), fast_math);
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
TEST_SUITE_END()

TEST_SUITE(WinogradLayer)
template <typename T>
using NEWinogradConvolutionLayerFixture = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, T>;

template <typename T>
using NEWinogradConvolutionLayerNoBiasFixture = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, T, false>;

TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(framework::dataset::concat(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                                                          datasets::SmallWinogradConvolutionLayer5x5Dataset()),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallNoBias, NEWinogradConvolutionLayerNoBiasFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(framework::dataset::concat(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                                                          datasets::SmallWinogradConvolutionLayer5x5Dataset()),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),

                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(GEMMConvolutionLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallConvolutionLayerDataset(), datasets::LargeConvolutionLayerDataset()),
                                                                           CNNDataTypes),
                                                                   framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) })),
input_shape, weights_shape, bias_shape, output_shape, info, dilation, data_type, act_info)
{
    // Set fixed point position data type allowed
    int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    auto bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

    // Create tensors
    Tensor src     = create_tensor<Tensor>(input_shape, data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));
    Tensor weights = create_tensor<Tensor>(weights_shape, data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));
    Tensor bias    = create_tensor<Tensor>(bias_shape, bias_data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));
    Tensor dst     = create_tensor<Tensor>(output_shape, data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    const QuantizationInfo src_quantization_info     = src.info()->quantization_info();
    const QuantizationInfo weights_quantization_info = weights.info()->quantization_info();

    // Create and configure function
    NEGEMMConvolutionLayer conv;
    conv.configure(&src, &weights, &bias, &dst, info, WeightsInfo(), dilation, act_info);

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
using NEGEMMConvolutionLayerFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolutionLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                 framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                 framework::dataset::make("DataType", DataType::F16)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                 ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::F16)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                               ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                  framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                  ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeConvolutionLayerDataset(),
                                                                                                                        framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NEGEMMConvolutionLayerFixedPointFixture = ConvolutionValidationFixedPointFixture<Tensor, Accessor, NEGEMMConvolutionLayer, T>;

TEST_SUITE(FixedPoint)
TEST_SUITE(QS8)
// We test for fixed point precision [4,6]
FIXTURE_DATA_TEST_CASE(RunTiny, NEGEMMConvolutionLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::TinyConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::QS8)),
                       framework::dataset::make("FractionalBits", 4, 7)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::QS8)),
                       framework::dataset::make("FractionalBits", 4, 7)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14)
FIXTURE_DATA_TEST_CASE(RunTiny, NEGEMMConvolutionLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::TinyConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NEGEMMConvolutionLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<Tensor, Accessor, NEGEMMConvolutionLayer, T>;

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                       QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                       QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
