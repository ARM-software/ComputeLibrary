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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DepthwiseConvolutionLayerDataset.h"
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
using namespace arm_compute::misc::shape_calculator;

namespace
{
constexpr RelativeTolerance<float>   tolerance_f32(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1); /**< Tolerance value for comparing reference's output against implementation's output for DataType::QASYMM8 */

const auto depth_multipliers = framework::dataset::make("DepthMultiplier", { 1, 2, 3 });
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DepthwiseConvLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate3x3, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Mismatching data type input/weights
                                                       TensorInfo(TensorShape(32U, 18U, 3U), 1, DataType::F32),     // Mismatching input feature maps
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Unsupported weights dimensions
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Mismatching depth multiplier
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::QASYMM8), // Invalid stride
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid biases size
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid biases dimensions
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid output size
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                     }),
               framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F16),
                                                         TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(5U, 5U, 2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::QASYMM8),
                                                         TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                                                       })),
               framework::dataset::make("BiasesInfo", { TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(2U), 1, DataType::S32),
                                                        TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(2U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                      })),
               framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::QASYMM8),
                                                        TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                      })),
               framework::dataset::make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
                                                      PadStrideInfo(1, 1, 0, 0),
                                                      PadStrideInfo(1, 1, 0, 0),
                                                      PadStrideInfo(1, 1, 0, 0),
                                                      PadStrideInfo(4, 1, 0, 0),
                                                      PadStrideInfo(1, 1, 0, 0),
                                                      PadStrideInfo(1, 1, 0, 0),
                                                      PadStrideInfo(1, 1, 0, 0),
                                                      PadStrideInfo(1, 1, 0, 0),
                                                     })),
               framework::dataset::make("DepthMultiplier", { 1,
                                                             1,
                                                             1,
                                                             3,
                                                             1,
                                                             1,
                                                             1,
                                                             1,
                                                             1,
                                                            })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, true })),
               input_info, weights_info, biases_info, output_info, conv_info, depth_multiplier, expected)
{
    bool is_valid = bool(NEDepthwiseConvolutionLayer3x3::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, depth_multiplier));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(ValidateGeneric, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
                framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching data type input/weights
                                                        TensorInfo(TensorShape(27U, 13U, 3U), 1, DataType::F32),    // Mismatching input feature maps
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching depth multiplier
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Invalid biases size
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Invalid biases dimensions
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Invalid output size
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
                                                          TensorInfo(TensorShape(3U, 3U, 24U), 1, DataType::QASYMM8),
                                                        })),
                framework::dataset::make("BiasesInfo", { TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U, 2U), 1, DataType::F32),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32),
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
                                                         TensorInfo(TensorShape(32U, 11U, 24U), 1, DataType::QASYMM8),
                                                       })),
                framework::dataset::make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
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
                                                              3,
                                                             })),
                framework::dataset::make("Expected", { false, false, false, false, false, false, true, true })),
                input_info, weights_info, biases_info, output_info, conv_info, depth_multiplier, expected)
{
    bool is_valid = bool(NEDepthwiseConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, depth_multiplier));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(F32)
TEST_SUITE(Generic)
template <typename T>
using NEDepthwiseConvolutionLayerFixture = DepthwiseConvolutionLayerValidationFixture<Tensor, Accessor, NEDepthwiseConvolutionLayer, T>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                       depth_multipliers),
                                                                                                                       framework::dataset::make("DataType",
                                                                                                                               DataType::F32)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                                                     depth_multipliers),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

TEST_SUITE(W3x3)
template <typename T>
using NEDepthwiseConvolutionLayerFixture3x3 = DepthwiseConvolutionLayerValidationFixture<Tensor, Accessor, NEDepthwiseConvolutionLayer3x3, T>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixture3x3<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                    depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F32)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixture3x3<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                        depth_multipliers),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F32)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunOptimized, NEDepthwiseConvolutionLayerFixture3x3<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::OptimizedDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                        framework::dataset::make("DepthMultiplier", 1)),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F32)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()

template <typename T>
using NEDepthwiseConvolutionLayerQuantizedFixture3x3 = DepthwiseConvolutionLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthwiseConvolutionLayer3x3, T>;
template <typename T>
using NEDepthwiseConvolutionLayerQuantizedFixture = DepthwiseConvolutionLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthwiseConvolutionLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(), depth_multipliers),
                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
