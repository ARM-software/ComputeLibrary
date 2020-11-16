/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLDepthwiseConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DepthwiseConvolutionLayerDataset.h"
#include "tests/datasets/DilatedDepthwiseConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DepthwiseConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<half_float::half>  tolerance_f16(half_float::half(0.01)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr RelativeTolerance<float>   tolerance_f32(0.01f);                  /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(0);                  /**< Tolerance value for comparing reference's output against implementation's output for DataType::QASYMM8 */
constexpr float                      tolerance_num = 0.05f;                 /**< Tolerance number */

const auto depth_multipliers       = framework::dataset::make("DepthMultiplier", { 1, 2, 5 });
const auto large_depth_multipliers = framework::dataset::make("DepthMultiplier", { 1, 2, 5, 8 });

//Activation Functions
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f, 0.f)
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DepthwiseConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
                framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching data type input/weights
                                                        TensorInfo(TensorShape(27U, 13U, 3U), 1, DataType::F32),    // Mismatching input feature maps
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching depth multiplier
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Invalid biases size
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Invalid biases dimensions
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Invalid output size
                                                        TensorInfo(TensorShape(27U, 13U, 8U), 1, DataType::F32),    // patch size bigger than input width
                                                        TensorInfo(TensorShape(27U, 13U, 8U), 1, DataType::F32),    // dilation < 1
                                                        TensorInfo(TensorShape(27U, 13U, 8U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 8U), 1, DataType::QASYMM8),
                                                      }),
                framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F16),
                                                          TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(3U, 3U, 24U), 1, DataType::QASYMM8),
                                                        })),
                framework::dataset::make("BiasesInfo", { TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32),
                                                       })),
                framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(32U, 11U, 24U), 1, DataType::QASYMM8),
                                                       })),
                framework::dataset::make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 1, 0),
                                                      })),
                framework::dataset::make("DepthMultiplier", { 1,
                                                              1,
                                                              3,
                                                              1,
                                                              1,
                                                              1,
                                                              2,
                                                              2,
                                                              2,
                                                              3,
                                                             })),
                framework::dataset::make("Dilation", { Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(20U, 1U),
                                                              Size2D(0U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                             })),
                framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, true, true })),
                input_info, weights_info, biases_info, output_info, conv_info, depth_multiplier, dilation, expected)
{
    bool is_valid = bool(CLDepthwiseConvolutionLayer::validate(&input_info.clone()->set_is_resizable(true), &weights_info.clone()->set_is_resizable(true), &biases_info.clone()->set_is_resizable(true), &output_info.clone()->set_is_resizable(true), conv_info, depth_multiplier,ActivationLayerInfo(), dilation));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLDepthwiseConvolutionLayerFixture = DepthwiseConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
TEST_SUITE(W3x3)
TEST_SUITE(NCHW)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(framework::dataset::concat(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                                  datasets::SmallDepthwiseConvolutionLayerDataset3x3NCHW()),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", DataLayout::NCHW)),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                                                        depth_multipliers),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", DataLayout::NHWC)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", DataLayout::NHWC)),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                                                        depth_multipliers),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // W3x3

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                        depth_multipliers),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                        depth_multipliers),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
TEST_SUITE(W3x3)
TEST_SUITE(NCHW)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(framework::dataset::concat(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                                  datasets::SmallDepthwiseConvolutionLayerDataset3x3NCHW()),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                     large_depth_multipliers),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", DataLayout::NCHW)),
                                                                                                                     ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(), depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       large_depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NCHW
TEST_SUITE(NHWC)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", DataLayout::NHWC)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                     large_depth_multipliers),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", DataLayout::NHWC)),
                                                                                                                     ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE(Dilation)

FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", DataLayout::NHWC)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       large_depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", DataLayout::NHWC)),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // W3x3

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                 depth_multipliers),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                 ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                                                     large_depth_multipliers),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                     ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                 depth_multipliers),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                 ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       large_depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLDepthwiseConvolutionLayerQuantizedFixture = DepthwiseConvolutionLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T>;
template <typename T>
using CLDepthwiseConvolutionLayerQuantizedPerChannelFixture = DepthwiseConvolutionLayerValidationQuantizedPerChannelFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T, int8_t>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })), // NCHW is tested with int8
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                       large_depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.7f, 2) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.8, 1) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })), // NCHW is tested with int8
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                                                       large_depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(1.3f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.9f, 11) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                       large_depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                       large_depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.8, 1) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM8_PER_CHANNEL)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                               depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                               large_depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.7f, 2) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                               depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.8, 1) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                                                               large_depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.9f, 11) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                               depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                               large_depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                               depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                               large_depth_multipliers),
                                                                       framework::dataset::make("SrcDataType", DataType::QASYMM8)),
                                                               framework::dataset::make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3
TEST_SUITE_END() // QSYMM8_PER_CHANNEL
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // DepthwiseConvolutionLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
