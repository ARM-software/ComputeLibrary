/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
using namespace arm_compute::misc::shape_calculator;

namespace
{
constexpr RelativeTolerance<float>   tolerance_f32(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1); /**< Tolerance value for comparing reference's output against implementation's output for DataType::QASYMM8 */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
RelativeTolerance<half_float::half> tolerance_f16(half_float::half(0.01)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                     tolerance_num = 0.05f;                 /**< Tolerance number */
#endif                                                                     // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

const auto depth_multipliers = framework::dataset::make("DepthMultiplier", { 1, 2, 5 });
const auto large_depth_multipliers = framework::dataset::make("DepthMultiplier", { 1, 2, 5, 8 });

//Activation Functions
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DepthwiseConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate3x3, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Mismatching data type input/weights
                                                       TensorInfo(TensorShape(32U, 18U, 3U), 1, DataType::F32),     // Mismatching input feature maps
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Unsupported weights dimensions
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Mismatching depth multiplier
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::QASYMM8), // Invalid stride
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid biases size
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid biases dimensions
                                                       TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid output size
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // patch size bigger than input width
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // dilation < 1
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
                                                        TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
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
                                                             1,
                                                             1,
                                                            })),
               framework::dataset::make("Dilation", { Size2D(1U, 1U),
                                                      Size2D(1U, 1U),
                                                      Size2D(1U, 1U),
                                                      Size2D(1U, 1U),
                                                      Size2D(1U, 1U),
                                                      Size2D(1U, 1U),
                                                      Size2D(1U, 1U),
                                                      Size2D(1U, 1U),
                                                      Size2D(25U, 1U),
                                                      Size2D(0U, 1U),
                                                      Size2D(1U, 1U),
                                                            })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, false, false, true })),
               input_info, weights_info, biases_info, output_info, conv_info, depth_multiplier,dilation, expected)
{
    bool is_valid = bool(NEDepthwiseConvolutionLayerOptimized::validate(&input_info.clone()->set_is_resizable(false),
     &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, depth_multiplier, ActivationLayerInfo(), dilation));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(ValidateGeneric, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
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
                                                       Size2D(25U, 1U),
                                                       Size2D(0U, 1U),
                                                       Size2D(1U, 1U),
                                                       Size2D(1U, 1U),
                                                             })),
                framework::dataset::make("Expected", { false, false, false, false, false, false,false, false, true, true })),
                input_info, weights_info, biases_info, output_info, conv_info, depth_multiplier,dilation, expected)
{
    bool is_valid = bool(NEDepthwiseConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, depth_multiplier, ActivationLayerInfo(), dilation));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(F32)
TEST_SUITE(Generic)
template <typename T>
using NEDepthwiseConvolutionLayerFixture = DepthwiseConvolutionLayerValidationFixture<Tensor, Accessor, NEDepthwiseConvolutionLayer, T>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                       depth_multipliers),
                                                                                                                       framework::dataset::make("DataType",
                                                                                                                               DataType::F32)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                       ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                                                     large_depth_multipliers),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                     ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                 depth_multipliers),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                 ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                     large_depth_multipliers),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                     ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic

template <typename T>
using NEDepthwiseConvolutionLayerFixtureOptimized = DepthwiseConvolutionLayerValidationFixture<Tensor, Accessor, NEDepthwiseConvolutionLayerOptimized, T>;

TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixtureOptimized<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                       depth_multipliers),
                       framework::dataset::make("DataType",
                                                DataType::F32)),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                       ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixtureOptimized<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                       large_depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixtureOptimized<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixtureOptimized<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       large_depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3

TEST_SUITE(Optimized)
FIXTURE_DATA_TEST_CASE(RunSmall3x3, NEDepthwiseConvolutionLayerFixtureOptimized<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                                       framework::dataset::make("DepthMultiplier", 1)),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmall5x5, NEDepthwiseConvolutionLayerFixtureOptimized<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                                       framework::dataset::make("DepthMultiplier", 1)),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge3x3, NEDepthwiseConvolutionLayerFixtureOptimized<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                                       framework::dataset::make("DepthMultiplier", 1)),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Optimized
TEST_SUITE_END() // F32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
TEST_SUITE(Generic)
template <typename T>
using NEDepthwiseConvolutionLayerFixture = DepthwiseConvolutionLayerValidationFixture<Tensor, Accessor, NEDepthwiseConvolutionLayer, T>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                      depth_multipliers),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::F16)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                      ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16, tolerance_num);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                        depth_multipliers),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                    large_depth_multipliers),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                    ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // Dilation

TEST_SUITE_END() // Generic
template <typename T>
using NEDepthwiseConvolutionLayerFixtureOptimized = DepthwiseConvolutionLayerValidationFixture<Tensor, Accessor, NEDepthwiseConvolutionLayerOptimized, T>;
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixtureOptimized<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                       depth_multipliers),
                       framework::dataset::make("DataType",
                                                DataType::F16)),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                       ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixtureOptimized<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                       large_depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16);
}

TEST_SUITE(Dilation)

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerFixtureOptimized<half>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerFixtureOptimized<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                       large_depth_multipliers),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16);
}

TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3

TEST_SUITE(Optimized)
FIXTURE_DATA_TEST_CASE(RunSmallW3x3, NEDepthwiseConvolutionLayerFixtureOptimized<half>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                                       framework::dataset::make("DepthMultiplier", 1)),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmallW5x5, NEDepthwiseConvolutionLayerFixtureOptimized<half>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                                       framework::dataset::make("DepthMultiplier", 1)),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLargeW3x3, NEDepthwiseConvolutionLayerFixtureOptimized<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                                       framework::dataset::make("DepthMultiplier", 1)),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // Optimized
TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE_END() // Float

template <typename T>
using NEDepthwiseConvolutionLayerQuantizedFixtureOptimized = DepthwiseConvolutionLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthwiseConvolutionLayerOptimized, T>;
template <typename T>
using NEDepthwiseConvolutionLayerQuantizedFixture = DepthwiseConvolutionLayerValidationQuantizedFixture<Tensor, Accessor, NEDepthwiseConvolutionLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                       depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.8f, 1) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                                                       large_depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.9f, 11) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerQuantizedFixtureOptimized<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(), depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerQuantizedFixtureOptimized<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                       large_depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE(Dilation)

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerQuantizedFixtureOptimized<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(), depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.7f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerQuantizedFixtureOptimized<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                       large_depth_multipliers),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3

TEST_SUITE(Optimized)
FIXTURE_DATA_TEST_CASE(RunSmall3x3, NEDepthwiseConvolutionLayerQuantizedFixtureOptimized<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                                                       framework::dataset::make("DepthMultiplier", 1)),
                                                               framework::dataset::make("DataType",
                                                                                        DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmall5x5, NEDepthwiseConvolutionLayerQuantizedFixtureOptimized<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                                                       framework::dataset::make("DepthMultiplier", 1)),
                                                               framework::dataset::make("DataType",
                                                                                        DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge3x3, NEDepthwiseConvolutionLayerQuantizedFixtureOptimized<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                                                       framework::dataset::make("DepthMultiplier", 1)),
                                                               framework::dataset::make("DataType",
                                                                                        DataType::QASYMM8)),
                                                       framework::dataset::make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                               framework::dataset::make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Optimized
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // DepthwiseConvLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
