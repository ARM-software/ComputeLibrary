/*
 * Copyright (c) 2017-2024 Arm Limited.
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

using framework::dataset::make;

namespace
{
RelativeTolerance<half_float::half>  tolerance_f16(half_float::half(0.01)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr RelativeTolerance<float>   tolerance_f32(0.01f);                  /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(0);                  /**< Tolerance value for comparing reference's output against implementation's output for DataType::QASYMM8 */
constexpr float                      tolerance_num = 0.05f;                 /**< Tolerance number */

const auto depth_multipliers       = make("DepthMultiplier", { 1, 4 });
const auto large_depth_multipliers = make("DepthMultiplier", { 2, 5, 8 });

// Activation Functions
const auto NoActivation = make("ActivationInfo", ActivationLayerInfo());

const auto ActivationFunctionsSmallDataset = make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 2.f, 0.f)
});

const auto ActivationFunctionsDataset = make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.8f, -0.5f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SOFT_RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SQUARE),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::HARD_SWISH),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 2.f, 1.f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::GELU)
});

const auto ActivationFunctionsQuantizedSmallDataset = make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 2.f, 0.f)
});

const auto ActivationFunctionsQuantizedDataset = make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 2.3f, -1.5f),
});

const auto IgnoredQuantizationInfo = make("IgnoredQuantizationInfo", QuantizationInfo());

} // namespace

TEST_SUITE(CL)
TEST_SUITE(DepthwiseConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
                make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching data type input/weights
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
                make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F16),
                                TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
                                TensorInfo(TensorShape(3U, 3U, 24U), 1, DataType::QASYMM8),
                        }),
                make("BiasesInfo", { TensorInfo(TensorShape(2U), 1, DataType::F32),
                                TensorInfo(TensorShape(2U), 1, DataType::F32),
                                TensorInfo(TensorShape(2U), 1, DataType::F32),
                                TensorInfo(TensorShape(4U), 1, DataType::F32),
                                TensorInfo(TensorShape(2U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(2U), 1, DataType::F32),
                                TensorInfo(TensorShape(16U), 1, DataType::F32),
                                TensorInfo(TensorShape(16U), 1, DataType::F32),
                                TensorInfo(TensorShape(16U), 1, DataType::F32),
                                TensorInfo(TensorShape(24U), 1, DataType::S32),
                        }),
                make("OutputInfo", { TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
                                TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
                                TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
                                TensorInfo(TensorShape(32U, 11U, 24U), 1, DataType::QASYMM8),
                        }),
                make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 0, 0),
                                PadStrideInfo(1, 1, 1, 0),
                                }),
                make("DepthMultiplier", { 1,
                                        1,
                                        3,
                                        1,
                                        1,
                                        1,
                                        2,
                                        2,
                                        2,
                                        3,
                                }),
                make("Dilation", { Size2D(1U, 1U),
                                Size2D(1U, 1U),
                                Size2D(1U, 1U),
                                Size2D(1U, 1U),
                                Size2D(1U, 1U),
                                Size2D(1U, 1U),
                                Size2D(20U, 1U),
                                Size2D(0U, 1U),
                                Size2D(1U, 1U),
                                Size2D(1U, 1U),
                                }),
                make("Expected", { false, false, false, false, false, false, false, false, true, true })),
                input_info, weights_info, biases_info, output_info, conv_info, depth_multiplier, dilation, expected)
{
    bool is_valid = bool(CLDepthwiseConvolutionLayer::validate(&input_info.clone()->set_is_resizable(true), &weights_info.clone()->set_is_resizable(true), &biases_info.clone()->set_is_resizable(true), &output_info.clone()->set_is_resizable(true), conv_info, depth_multiplier,ActivationLayerInfo(), dilation));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLDepthwiseConvolutionLayerFixture = DepthwiseConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T>;
template <typename T>
using CLDepthwiseConvolutionLayerMixedDataLayoutFixture = DepthwiseConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T, true>;
template <typename T>
using CLDepthwiseConvolutionLayerInPlaceFixture = DepthwiseConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T, false, true>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
TEST_SUITE(W3x3)
TEST_SUITE(NCHW)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL,
    combine(
        framework::dataset::concat(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                        datasets::SmallDepthwiseConvolutionLayerDataset3x3NCHW()),
        depth_multipliers,
        make("DataType", DataType::F16),
        make("DataLayout", DataLayout::NCHW),
        ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL,
    combine(
        datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
        depth_multipliers,
        make("DataType", DataType::F16),
        make("DataLayout", { DataLayout::NCHW }),
        ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL,
    combine(
        datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
        depth_multipliers,
        make("DataType", DataType::F16),
        make("DataLayout", DataLayout::NHWC),
        ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
    combine(
        datasets::LargeDepthwiseConvolutionLayerDataset3x3Fp16Subset(),
        large_depth_multipliers,
        make("DataType", DataType::F16),
        make("DataLayout", DataLayout::NHWC),
        make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                                                    depth_multipliers),
                                                                                                                    make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    make("DataLayout", { DataLayout::NHWC })),
                                                                                                                   ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3Fp16Subset(),
                                                           large_depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F16)),
                                           make("DataLayout", { DataLayout::NHWC })),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // W3x3

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                    depth_multipliers),
                                                                                                                    make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDatasetFp16Subset(),
                                                                                                                        large_depth_multipliers),
                                                                                                                        make("DataType",
                                                                                                                                DataType::F16)),
                                                                                                                        make("DataLayout", { DataLayout::NHWC })),
                                                                                                                        make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}

FIXTURE_DATA_TEST_CASE_NEW(RunActivations, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
    combine(
        make("In", TensorShape(33U, 27U, 11U, 3U)),
        make("Weights", Size2D(3U, 4U)),
        make("Info", PadStrideInfo(1, 2, 0, 1)),
        make("Dilation", Size2D(2U, 2U)),
        make("DepthMultiplier", { 2 }),
        make("DataType", DataType::F16),
        make("DataLayout", { DataLayout::NHWC }),
        ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                    depth_multipliers),
                                                                                                                    make("DataType",
                                                                                                                            DataType::F16)),
                                                                                                                    make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDatasetFp16Subset(),
                                                           large_depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F16)),
                                           make("DataLayout", { DataLayout::NHWC })),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic

TEST_SUITE(InPlace)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerInPlaceFixture<half>, framework::DatasetMode::ALL,
                           combine(combine(combine(combine(datasets::SmallInPlaceDepthwiseConvolutionLayerDataset(),
                                                           make("DepthMultiplier", { 1 })),
                                                   make("DataType",
                                                                            DataType::F16)),
                                           make("DataLayout", { DataLayout::NHWC })),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_src), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // InPlace
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
TEST_SUITE(W3x3)
TEST_SUITE(NCHW)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                           combine(combine(combine(combine(framework::dataset::concat(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                                      datasets::SmallDepthwiseConvolutionLayerDataset3x3NCHW()),
                                                           depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", DataLayout::NCHW)),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                           combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(), depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", DataLayout::NCHW)),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                           combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                           depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", DataLayout::NHWC)),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunMixedDataLayout, CLDepthwiseConvolutionLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT,
                           combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                           make("DepthMultiplier", { 2 })),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", DataLayout::NHWC)),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                           large_depth_multipliers),
                           make("DataType",
                                                    DataType::F32)),
                           make("DataLayout", DataLayout::NHWC)),
                           make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(Dilation)

FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                           combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                           depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", DataLayout::NHWC)),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                           large_depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", DataLayout::NHWC)),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // W3x3

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                     depth_multipliers),
                                                                                                                     make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                    ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                           large_depth_multipliers),
                           make("DataType",
                                                    DataType::F32)),
                           make("DataLayout", { DataLayout::NHWC })),
                           make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE_NEW(RunLargeKernelSize, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL,
                           combine(combine(combine(combine(datasets::LargeKernelSizeDepthwiseConvolutionLayerNHWCDataset(),
                                                           make("DepthMultiplier", { 1 })),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", { DataLayout::NHWC })),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE_NEW(RunActivations, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
    combine(
        make("In", TensorShape(33U, 27U, 11U, 3U)),
        make("Weights", Size2D(3U, 4U)),
        make("Info", PadStrideInfo(1, 2, 0, 1)),
        make("Dilation", Size2D(2U, 2U)),
        make("DepthMultiplier", { 2 }),
        make("DataType", DataType::F32),
        make("DataLayout", { DataLayout::NHWC }),
        ActivationFunctionsDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                     depth_multipliers),
                                                                                                                     make("DataType",
                                                                                                                             DataType::F32)),
                                                                                                                     make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                    ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                           large_depth_multipliers),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", { DataLayout::NHWC })),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic

TEST_SUITE(InPlace)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerInPlaceFixture<float>, framework::DatasetMode::ALL,
                           combine(combine(combine(combine(datasets::SmallInPlaceDepthwiseConvolutionLayerDataset(),
                                                           make("DepthMultiplier", { 1 })),
                                                   make("DataType",
                                                                            DataType::F32)),
                                           make("DataLayout", { DataLayout::NHWC })),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_src), _reference, tolerance_f32);
}
TEST_SUITE_END() // InPlace
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLDepthwiseConvolutionLayerQuantizedFixture = DepthwiseConvolutionLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T>;
template <typename T>
using CLDepthwiseConvolutionLayerQuantizedMixedDataLayoutFixture = DepthwiseConvolutionLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T, true>;
template <typename T>
using CLDepthwiseConvolutionLayerQuantizedPerChannelFixture = DepthwiseConvolutionLayerValidationQuantizedPerChannelFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T, int8_t>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NHWC }), // NCHW is tested with int8
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 128), QuantizationInfo(2.2f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(1.f, 128) }),
        make("DataLayout", { DataLayout::NHWC }), // NCHW is tested with int8
        ActivationFunctionsQuantizedSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
    combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
        large_depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NHWC }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunActivations, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
    combine(
        make("In", TensorShape(33U, 27U, 11U, 3U)),
        make("Weights", Size2D(3U, 4U)),
        make("Info", PadStrideInfo(1, 2, 0, 1)),
        make("Dilation", Size2D(2U, 2U)),
        make("DepthMultiplier", { 2U }),
        make("DataType", DataType::QASYMM8),
        make("SrcQuantizationInfo", { QuantizationInfo(2.2f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.1f, 128) }),
        make("DataLayout", { DataLayout::NHWC }),
        ActivationFunctionsQuantizedDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NHWC }), // NCHW is tested with int8
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.8, 1) }),
        make("DataLayout", { DataLayout::NHWC }), // NCHW is tested with int8
        ActivationFunctionsQuantizedSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
    combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
        large_depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NHWC }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NHWC }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10), QuantizationInfo(2.2f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) }),
        make("DataLayout", { DataLayout::NHWC }),
        ActivationFunctionsQuantizedSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
    combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
        large_depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NHWC }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8),
        make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) }),
        make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }),
        ActivationFunctionsQuantizedSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunMixedDataLayout, CLDepthwiseConvolutionLayerQuantizedMixedDataLayoutFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
        make("DepthMultiplier", { 2 }),
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
    combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
        large_depth_multipliers,
        make("DataType", DataType::QASYMM8),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NHWC }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8_SIGNED),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NCHW }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation, CLDepthwiseConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8_SIGNED),
        make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10), QuantizationInfo(2.2f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) }),
        make("DataLayout", { DataLayout::NCHW }),
        ActivationFunctionsQuantizedSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunMixedDataLayout, CLDepthwiseConvolutionLayerQuantizedMixedDataLayoutFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
        make("DepthMultiplier", { 2 }),
        make("DataType", DataType::QASYMM8_SIGNED),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NCHW }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunActivations, CLDepthwiseConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY,
    combine(
        make("In", TensorShape(33U, 27U, 11U, 3U)),
        make("Weights", Size2D(3U, 4U)),
        make("Info", PadStrideInfo(1, 2, 0, 1)),
        make("Dilation", Size2D(2U, 2U)),
        make("DepthMultiplier", { 2U }),
        make("DataType", DataType::QASYMM8_SIGNED),
        make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) }),
        make("DataLayout", { DataLayout::NHWC }),
        ActivationFunctionsQuantizedDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8_SIGNED),
        IgnoredQuantizationInfo,
        IgnoredQuantizationInfo,
        make("DataLayout", { DataLayout::NCHW }),
        NoActivation))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation, CLDepthwiseConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
        depth_multipliers,
        make("DataType", DataType::QASYMM8_SIGNED),
        make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10), QuantizationInfo(2.2f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.8, 1) }),
        make("DataLayout", { DataLayout::NCHW }),
        ActivationFunctionsQuantizedSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM8_PER_CHANNEL)
TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                           combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                   depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) })),
                                           make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                   large_depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.7f, 2) })),
                                           make("DataLayout", { DataLayout::NHWC })),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunActivations, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<int8_t>, framework::DatasetMode::NIGHTLY,
    combine(
        make("In", TensorShape(33U, 27U, 11U, 3U)),
        make("Weights", Size2D(3U, 4U)),
        make("Info", PadStrideInfo(1, 2, 0, 1)),
        make("Dilation", Size2D(2U, 2U)),
        make("DepthMultiplier", { 2U }),
        make("SrcDataType", DataType::QASYMM8_SIGNED),
        make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL),
        make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) }),
        make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) }),
        make("DataLayout", { DataLayout::NHWC }),
        ActivationFunctionsQuantizedDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                           combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                   depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.8, 1) })),
                                           make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                                                                   large_depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.9f, 11) })),
                                           make("DataLayout", { DataLayout::NHWC })),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                           combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                                   depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.3f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.5f, 4) })),
                                           make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                   large_depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                           make("DataLayout", { DataLayout::NHWC })),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                           combine(combine(combine(combine(combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                   depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                           make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                  ActivationFunctionsSmallDataset))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                           combine(combine(combine(combine(combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                   large_depth_multipliers),
                                                                           make("SrcDataType", DataType::QASYMM8)),
                                                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL)),
                                                           make("SrcQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                                   make("DstQuantizationInfo", { QuantizationInfo(0.5f, 10) })),
                                           make("DataLayout", { DataLayout::NHWC })),
                                   make("ActivationInfo", ActivationLayerInfo())))
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
