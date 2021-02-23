/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMMConv2d.h"
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
namespace detail
{
template <>
void configure_conv_function<NEGEMMConv2d, Tensor>(NEGEMMConv2d &func,
                                                   Tensor *src, const Tensor *weights, const Tensor *bias, Tensor *dst,
                                                   const PadStrideInfo &info, const WeightsInfo &weights_info,
                                                   const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_UNUSED(weights_info);

    Conv2dInfo conv_info(info, dilation, act_info, false, num_groups);
    func.configure(src, weights, bias, dst, conv_info);
}
} // namespace detail
namespace
{
const RelativeTolerance<float> rel_tolerance_f32(0.01f);              /**< Relative tolerance for FP32 types */
const RelativeTolerance<float> rel_tolerance_winograd_3x3_f32(0.05f); /**< Relative tolerance for FP32 types */
const AbsoluteTolerance<float> abs_tolerance_f32(0.002f);             /**< Absolute tolerance for FP32 types */
const AbsoluteTolerance<float> abs_tolerance_1xN_f32(0.0041f);        /**< Absolute tolerance for FP32 types */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const AbsoluteTolerance<half> tolerance_convolution_layer_f16(half(0.4f));
constexpr float               tolerance_num_f16 = 0.15f;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2f)); /**< Relative tolerance value for FP16 types */
const AbsoluteTolerance<float>            abs_tolerance_f16(0.2f);                   /**< Absolute tolerance for FP16 types */
constexpr float                           tolerance_num = 0.07f;                     /**< Tolerance number for the FP16 implementation */
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
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f)
});

const auto QuantizationData = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(0.5f, 10),
    QuantizationInfo(0.3f, 3),
    QuantizationInfo(1.f, 10),
    QuantizationInfo(1.1f, 10),
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
                                          framework::dataset::make("InputInfo", { TensorInfo(TensorShape(18U, 18U, 32U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(23U, 27U, 32U, 4U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(3U, 3U, 2U, 1U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32)
                                          }),
                                          framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 32U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 32U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16)
                                          })),
                                          framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(16U, 16U, 21U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(19U, 23U, 21U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32)
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
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // ConvolutionLayer

TEST_SUITE(WinogradLayer)
template <typename T>
using NEWinogradConvolutionLayerFixture = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, T>;

template <typename T>
using NEWinogradConvolutionLayerNoBiasFixture = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, T, T, false>;

TEST_SUITE(FP32)

TEST_SUITE(Conv1x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv1x3

TEST_SUITE(Conv3x1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv3x1

TEST_SUITE(Conv1x5)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv1x5

TEST_SUITE(Conv5x1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv5x1

TEST_SUITE(Conv7x1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer7x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer7x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}
TEST_SUITE_END() // Conv7x1

TEST_SUITE(Conv1x7)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x7Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer7x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}
TEST_SUITE_END() // Conv1x7

TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    // floating point arithmetic the Winograd results will not be exactly the same as direct convolution, especially for big shapes
    validate(Accessor(_target), _reference, rel_tolerance_winograd_3x3_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END() // Conv3x3

TEST_SUITE(Conv5x5)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

TEST_SUITE_END() // Conv5x5

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

TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
using CLWinogradConvolutionLayerFastMathFixture16 = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, half, float>;

TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}
TEST_SUITE_END() // Conv3x3
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END() // WinogradLayer

TEST_SUITE(GEMMConvolutionLayer)
template <typename T>
using NEGEMMConvolutionLayerFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolutionLayer, T>;

TEST_SUITE(Float)
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16)
TEST_SUITE(BFLOAT16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                    framework::dataset::make("DataType", DataType::BFLOAT16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                            ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END() // BFLOAT16
#endif           /* defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16) */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                           ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                            ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using NEGEMMConvolutionLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<Tensor, Accessor, NEConvolutionLayer, T>;

template <typename T>
using NEGEMMConvolutionLayerQuantizedPerChannelFixture = ConvolutionValidationQuantizedPerChannelFixture<Tensor, Accessor, NEConvolutionLayer, T, int8_t>;

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                       QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                      framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.01f, -10) })),
                                                                                                                      QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM8_PER_CHANNEL)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8 })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmallSigned, NEGEMMConvolutionLayerQuantizedPerChannelFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8_SIGNED })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QSYMM8_PER_CHANNEL
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // GEMMConvolutionLayer

TEST_SUITE(DirectGEMMConv2d)
template <typename T>
using NEDirectGEMMConv2dLayerFixture = ConvolutionValidationFixture<Tensor, Accessor, NEGEMMConv2d, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectGEMMConv2dLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                     framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                     framework::dataset::make("DataType", DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                             ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

#ifdef __aarch64__
template <typename T>
using NEDirectGEMMConv2dLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<Tensor, Accessor, NEGEMMConv2d, T>;

template <typename T>
using NEDirectGEMMConv2dLayerQuantizedPerChannelFixture = ConvolutionValidationQuantizedPerChannelFixture<Tensor, Accessor, NEGEMMConv2d, T, int8_t>;

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectGEMMConv2dLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                        framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                        framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                        QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectGEMMConv2dLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.01f, -10) })),
                                                                                                                       QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM8_PER_CHANNEL)
FIXTURE_DATA_TEST_CASE(RunSmallSigned, NEDirectGEMMConv2dLayerQuantizedPerChannelFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8_SIGNED })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QSYMM8_PER_CHANNEL
TEST_SUITE_END() // Quantized
#endif           // __aarch64__

TEST_SUITE_END() // DirectGEMMConv2d

TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
