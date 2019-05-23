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
#include "arm_compute/core/CL/kernels/CLWinogradFilterTransformKernel.h"
#include "arm_compute/core/CL/kernels/CLWinogradOutputTransformKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLWinogradInputTransform.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeConvolutionLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/datasets/WinogradInputTransformDataset.h"
#include "tests/datasets/WinogradOutputTransformDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WinogradConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// *INDENT-OFF*
// clang-format off
constexpr AbsoluteTolerance<float> tolerance_f32(0.002f);
const AbsoluteTolerance<half> tolerance_f16(half(0.5f));
constexpr AbsoluteTolerance<float> tolerance_convolution_layer_f32(0.1f);
const AbsoluteTolerance<half> tolerance_convolution_layer_f16(half(0.4f));
RelativeTolerance<half_float::half> rel_tolerance_f16(half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for FP16 data types */
constexpr float                     tolerance_num   = 0.05f;  /**< Tolerance number */
constexpr float                     abs_tolerance_convolution_layer_f16   = 2.5f;  /**< Tolerance number */

// Input transform
const auto SmallWinogradInputTransformDatasetNCHW =
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset2x2_3x3(),
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset2x1_3x1(),
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset1x2_1x3(),
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x4_3x3(),
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x1_3x1(),
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset1x4_1x3(),
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x4_5x5(),
           framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x1_5x1(),
                                      datasets::SmallWinogradInputTransformDataset1x4_1x5()))))))));

const auto SmallWinogradInputTransformDatasetNHWC = framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x4_3x3(),
                                                    framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x1_3x1(),
                                                    framework::dataset::concat(datasets::SmallWinogradInputTransformDataset1x4_1x3(),
                                                    framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x4_5x5(),
                                                    framework::dataset::concat(datasets::SmallWinogradInputTransformDataset4x1_5x1(),
                                                                               datasets::SmallWinogradInputTransformDataset1x4_1x5())))));

const auto SmallWinogradInputTransformDatasetNHWC_FP32 = framework::dataset::concat(SmallWinogradInputTransformDatasetNHWC,
                                                         framework::dataset::concat(datasets::SmallWinogradInputTransformDataset1x2_1x7(),
                                                         framework::dataset::concat(datasets::SmallWinogradInputTransformDataset2x1_7x1(),
                                                                                    datasets::SmallWinogradInputTransformDataset2x2_7x7())));

const auto LargeWinogradInputTransformDatasetNCHW =
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset2x2_3x3(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset2x1_3x1(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset1x2_1x3(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset4x4_3x3(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset4x1_3x1(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset1x4_1x3(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset4x4_5x5(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset4x1_5x1(),
                                      datasets::LargeWinogradInputTransformDataset1x4_1x5()))))))));

const auto LargeWinogradInputTransformDatasetNHWC =
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset4x4_3x3(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset4x4_5x5(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset4x1_5x1(),
                                      datasets::LargeWinogradInputTransformDataset1x4_1x5())));

const auto LargeWinogradInputTransformDatasetNHWC_FP32 =
           framework::dataset::concat(LargeWinogradInputTransformDatasetNHWC,
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset1x2_1x7(),
           framework::dataset::concat(datasets::LargeWinogradInputTransformDataset2x1_7x1(),
                                     (datasets::LargeWinogradInputTransformDataset2x2_7x7()))));

// Filter transform
const auto SmallWinogradFilterTransformDatasetNCHW =
           framework::dataset::concat(combine(datasets::Small3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 2U), Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Small3x1Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 1U), Size2D(4U, 1U) })),
           framework::dataset::concat(combine(datasets::Small1x3Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 2U), Size2D(1U, 4U) })),
           framework::dataset::concat(combine(datasets::Small5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Small5x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
                                      combine(datasets::Small1x5Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })))))));

const auto SmallWinogradFilterTransformDatasetNHWC_F16 =
           framework::dataset::concat(combine(datasets::Small3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Small3x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
           framework::dataset::concat(combine(datasets::Small1x3Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })),
           framework::dataset::concat(combine(datasets::Small5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Small5x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
                                     (combine(datasets::Small1x5Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) }))))))));

const auto SmallWinogradFilterTransformDatasetNHWC_F32 =
           framework::dataset::concat(SmallWinogradFilterTransformDatasetNHWC_F16,
           framework::dataset::concat(combine(datasets::Small7x7Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 2U) })),
           framework::dataset::concat(combine(datasets::Small7x1Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 1U) })),
                                      combine(datasets::Small1x7Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 2U) })))));

const auto LargeWinogradFilterTransformDatasetNCHW =
           framework::dataset::concat(combine(datasets::Large3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 2U), Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Large3x1Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 1U), Size2D(4U, 1U) })),
           framework::dataset::concat(combine(datasets::Large1x3Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 2U), Size2D(1U, 4U) })),
           framework::dataset::concat(combine(datasets::Large5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Large5x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
                                      combine(datasets::Large1x5Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })))))));

const auto LargeWinogradFilterTransformDatasetNHWC_F16 =
           framework::dataset::concat(combine(datasets::Large3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Large3x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
           framework::dataset::concat(combine(datasets::Large1x3Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })),
           framework::dataset::concat(combine(datasets::Large5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Large5x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
                                      combine(datasets::Large1x5Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })))))));

const auto LargeWinogradFilterTransformDatasetNHWC_F32 =
           framework::dataset::concat(LargeWinogradFilterTransformDatasetNHWC_F16,
           framework::dataset::concat(combine(datasets::Large7x7Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 2U) })),
           framework::dataset::concat(combine(datasets::Large7x1Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 1U) })),
                                      combine(datasets::Large1x7Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 2U) })))));

// Output transform
const auto SmallWinogradOutputTransformDatasetNCHW = datasets::SmallWinogradOutputTransformDatasetNCHW();

const auto SmallWinogradOutputTransformDatasetNHWC_F16 = datasets::SmallWinogradOutputTransformDatasetNHWC_F16();

const auto SmallWinogradOutputTransformDatasetNHWC_F32 = datasets::SmallWinogradOutputTransformDatasetNHWC_F32();

const auto LargeWinogradOutputTransformDatasetNCHW = datasets::LargeWinogradOutputTransformDatasetNCHW();

const auto LargeWinogradOutputTransformDatasetNHWC_F16 = datasets::LargeWinogradOutputTransformDatasetNHWC_F16();

const auto LargeWinogradOutputTransformDatasetNHWC_F32 = datasets::LargeWinogradOutputTransformDatasetNHWC_F32();

//Activation Functions
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU)
});
const auto ActivationFunctionsSmallDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SOFT_RELU)
});
} // namespace

using namespace arm_compute::misc::shape_calculator;

TEST_SUITE(CL)
TEST_SUITE(Winograd)

TEST_SUITE(InputTransform)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::F16),     // F16 not supported
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::QASYMM8), // QASYMM8 not supported
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::F32),     // Kernel size not supported
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::F32),     // Strides not supported
                                                                                        TensorInfo(TensorShape(53U, 33U, 4U), 1, DataType::F32),         // Padding needed
                                                                                        TensorInfo(TensorShape(34U, 42U, 7U, 3U), 1, DataType::F32),     // Padding needed
                                                                                        TensorInfo(TensorShape(31U, 37U, 37U), 1, DataType::F32)         // Padding needed
                                                                                    }),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(5U, 5U, 16U, 3U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(5U, 5U, 16U, 3U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(5U, 5U, 16U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(5U, 1U, 16U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 442U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U, 320U, 16U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(37U, 304U, 16U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("WinogradInfo", {
                                                                                        WinogradInfo(Size2D(2, 2), Size2D(3, 3), Size2D(53U, 21U), PadStrideInfo(1, 1, 1, 0), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2, 2), Size2D(3, 3), Size2D(53U, 21U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2, 2), Size2D(3, 3), Size2D(53U, 21U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2, 2), Size2D(3, 3), Size2D(53U, 21U), PadStrideInfo(2, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2, 2), Size2D(3, 3), Size2D(53U, 33U), PadStrideInfo(1, 1, 0, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2, 2), Size2D(3, 3), Size2D(34U, 42U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2, 2), Size2D(3, 3), Size2D(31U, 37U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW)
                                                                                    })),
                                                framework::dataset::make("Expected", { false, false, false, false, false, false, false })),
                                            input_info, output_info, winograd_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradInputTransform::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), winograd_info)) == expected, framework::LogLevel::ERRORS);
}

using CLWinogradInputTransformFixtureFP32 = WinogradInputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradInputTransform, float>;
using CLWinogradInputTransformFixtureFP16 = WinogradInputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradInputTransform, half>;

TEST_SUITE(NCHW)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixtureFP32, framework::DatasetMode::PRECOMMIT, combine(combine(SmallWinogradInputTransformDatasetNCHW,
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                     framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixtureFP32, framework::DatasetMode::NIGHTLY, combine(combine(LargeWinogradInputTransformDatasetNCHW,
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                   framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixtureFP16, framework::DatasetMode::PRECOMMIT, combine(combine(SmallWinogradInputTransformDatasetNCHW,
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                     framework::dataset::make("DataType", { DataType::F16 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixtureFP16, framework::DatasetMode::NIGHTLY, combine(combine(LargeWinogradInputTransformDatasetNCHW,
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                   framework::dataset::make("DataType", { DataType::F16 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixtureFP16, framework::DatasetMode::PRECOMMIT, combine(combine(SmallWinogradInputTransformDatasetNHWC,
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                     framework::dataset::make("DataType", { DataType::F16 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixtureFP16, framework::DatasetMode::NIGHTLY, combine(combine(LargeWinogradInputTransformDatasetNHWC,
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("DataType", { DataType::F16 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixtureFP32, framework::DatasetMode::PRECOMMIT, combine(combine(SmallWinogradInputTransformDatasetNHWC_FP32,
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                     framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixtureFP32, framework::DatasetMode::NIGHTLY, combine(combine(LargeWinogradInputTransformDatasetNHWC_FP32,
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // InputTransform

TEST_SUITE(FilterTransform)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 3U), 1, DataType::F16),     // F16 supported
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 3U), 1, DataType::QASYMM8), // QASYMM8 not supported
                                                                                        TensorInfo(TensorShape(5U, 5U, 5U, 3U), 1, DataType::F32),     // Kernel size not supported
                                                                                        TensorInfo(TensorShape(3U, 3U), 1, DataType::F32),             // Output tile not supported
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 3U), 1, DataType::F32),     // valid
                                                                                        TensorInfo(TensorShape(3U, 3U, 37U, 2U), 1, DataType::F32),    // valid
                                                                                        TensorInfo(TensorShape(3U, 3U, 37U, 22U), 1, DataType::F32)    // valid
                                                                                    }),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(1U, 1U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(2U, 37U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(22U, 37U, 36U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("WinogradInfo", {
                                                                                          WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW  /* Not needed */ ),
                                                                                          WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW  /* Not needed */ ),
                                                                                          WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW  /* Not needed */ ),
                                                                                          WinogradInfo(Size2D(3U, 3U), Size2D(3U, 3U), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW  /* Not needed */ ),
                                                                                          WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW  /* Not needed */ ),
                                                                                          WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW  /* Not needed */ ),
                                                                                          WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, DataLayout::NCHW  /* Not needed */ )
                                                                                         })),
                                                framework::dataset::make("Expected", { true, false, false, false, true, true, true })),
                                            input_info, output_info, winograd_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradFilterTransformKernel::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), winograd_info)) == expected, framework::LogLevel::ERRORS);
}

using CLWinogradFilterTransform        = CLSynthetizeFunctionWithZeroConstantBorder<CLWinogradFilterTransformKernel, 0>;
using CLWinogradFilterTransformFixtureFP32 = WinogradFilterTransformValidationFixture<CLTensor, CLAccessor, CLWinogradFilterTransform, float>;
using CLWinogradFilterTransformFixtureFP16 = WinogradFilterTransformValidationFixture<CLTensor, CLAccessor, CLWinogradFilterTransform, half>;

TEST_SUITE(NCHW)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixtureFP32, framework::DatasetMode::PRECOMMIT,
                       combine(combine(SmallWinogradFilterTransformDatasetNCHW,
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixtureFP32, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradFilterTransformDatasetNCHW,
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixtureFP16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(SmallWinogradFilterTransformDatasetNCHW,
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("DataType", { DataType::F16 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixtureFP16, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradFilterTransformDatasetNCHW,
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("DataType", { DataType::F16 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixtureFP16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(SmallWinogradFilterTransformDatasetNHWC_F16,
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                       framework::dataset::make("DataType", { DataType::F16 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixtureFP16, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradFilterTransformDatasetNHWC_F16,
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                       framework::dataset::make("DataType", { DataType::F16 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixtureFP32, framework::DatasetMode::PRECOMMIT,
                       combine(combine(SmallWinogradFilterTransformDatasetNHWC_F32,
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixtureFP32, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradFilterTransformDatasetNHWC_F32,
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // FilterTransform

TEST_SUITE(OutputTransform)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(512U, 49U, 16U, 5U), 1, DataType::F16),      // F16 supported
                                                                                        TensorInfo(TensorShape(512U, 49U, 16U, 5U), 1, DataType::QASYMM8),  // QASYMM8 not supported
                                                                                        TensorInfo(TensorShape(512U, 49U, 16U, 5U), 1, DataType::F32),      // Kernel size not supported
                                                                                        TensorInfo(TensorShape(512U, 49U, 16U, 5U), 1, DataType::F32),      // Valid
                                                                                        TensorInfo(TensorShape(13U, 108U, 16U, 4U), 1, DataType::F32),      // Padding needed
                                                                                        TensorInfo(TensorShape(7U, 20U, 16U, 7U), 1, DataType::F32),        // Valid
                                                                                        TensorInfo(TensorShape(7U, 20U, 16U, 7U), 1, DataType::F32),        // Wrong WinogradInfo
                                                                                        TensorInfo(TensorShape(7U, 256U, 36U, 3U), 1, DataType::F32),       // Valid
                                                                                        TensorInfo(TensorShape(7U, 256U, 16U, 3U), 1, DataType::F32)        // Wrong number of batches
                                                                                    }),
                                                framework::dataset::make("BiasInfo", {
                                                                                        TensorInfo(TensorShape(512U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(512U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(512U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(512U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(13U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(14U, 14U, 512U, 5U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(14U, 14U, 512U, 5U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(14U, 14U, 512U, 5U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(14U, 14U, 512U, 5U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(17U, 23U, 13U, 4U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(8U, 10U, 7U, 7U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U, 9U, 7U, 7U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(64U, 64U, 7U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(64U, 64U, 7U, 3U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("WinogradInfo", {
                                                                                        WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2U, 2U), Size2D(5U, 5U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(14U, 14U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(17U, 23U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2U, 2U), Size2D(3U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(2U, 3U), Size2D(3U, 3U), Size2D(8U, 10U), PadStrideInfo(1, 1, 0, 0), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(64U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW),
                                                                                        WinogradInfo(Size2D(4U, 4U), Size2D(3U, 3U), Size2D(64U, 64U), PadStrideInfo(1, 1, 1, 1), DataLayout::NCHW)
                                                                                    })),
                                                framework::dataset::make("Expected", { true, false, false, true, false, true, false, true, false })),
                                            input_info, bias_info, output_info, winograd_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradOutputTransformKernel::validate(&input_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), winograd_info)) == expected, framework::LogLevel::ERRORS);
}

using CLWinogradOutputTransform        = CLSynthetizeFunctionWithZeroConstantBorder<CLWinogradOutputTransformKernel, 0>;
using CLWinogradOutputTransformFixtureFP32 = WinogradOutputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradOutputTransform, float>;
using CLWinogradOutputTransformFixtureFP16 = WinogradOutputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradOutputTransform, half>;

TEST_SUITE(NCHW)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradOutputTransformFixtureFP16, framework::DatasetMode::ALL,
                       combine(combine(SmallWinogradOutputTransformDatasetNCHW,
                               framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradOutputTransformFixtureFP16, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradOutputTransformDatasetNCHW,
                               framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradOutputTransformFixtureFP32, framework::DatasetMode::ALL,
                       combine(combine(SmallWinogradOutputTransformDatasetNCHW,
                               framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradOutputTransformFixtureFP32, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradOutputTransformDatasetNCHW,
                               framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradOutputTransformFixtureFP16, framework::DatasetMode::ALL,
                       combine(combine(SmallWinogradOutputTransformDatasetNHWC_F16,
                               framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradOutputTransformFixtureFP16, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradOutputTransformDatasetNHWC_F16,
                               framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradOutputTransformFixtureFP32, framework::DatasetMode::ALL,
                       combine(combine(SmallWinogradOutputTransformDatasetNHWC_F32,
                               framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradOutputTransformFixtureFP32, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradOutputTransformDatasetNHWC_F32,
                               framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("ActivationInfo",{ ActivationLayerInfo() }) ))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // OutputTransform

TEST_SUITE(ConvolutionLayer)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
                                                framework::dataset::make("InputInfo", {
                                                                                        TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F16),     // Insufficient padding
                                                                                        TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),     // Datatype mismatch
                                                                                        TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32), // Stride y not supported
                                                                                        TensorInfo(TensorShape(16U, 16U, 8U), 1, DataType::F32),     // Padding needed
                                                                                        TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32)  // Kernel size not supported
                                                                                      }),
                                                framework::dataset::make("WeightsInfo", {
                                                                                        TensorInfo(TensorShape(3U, 3U, 2U, 19U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(3U, 3U, 2U, 19U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(3U, 3U, 8U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16)
                                                                                        })),
                                                framework::dataset::make("BiasesInfo", {
                                                                                        TensorInfo(TensorShape(19U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(19U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(21U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(16U), 1, DataType::F32)
                                                                                       })),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(16U, 16U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32)
                                                                                       })),
                                                framework::dataset::make("ConvInfo", {
                                                                                        PadStrideInfo(1, 1, 1, 1),
                                                                                        PadStrideInfo(1, 1, 1, 1),
                                                                                        PadStrideInfo(1, 2, 0, 0),
                                                                                        PadStrideInfo(1, 1, 1, 1),
                                                                                        PadStrideInfo(1, 1, 1, 0)
                                                                                                                 })),
                                                framework::dataset::make("Expected", { false, false, false, false, false })),
               input_info, weights_info, bias_info, output_info, conv_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info)) == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE(FP32)
using CLWinogradConvolutionLayerFastMathFixture = WinogradConvolutionLayerFastMathValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, float>;
TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv3x3

TEST_SUITE(Conv3x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x1Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsSmallDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x1Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv3x1

TEST_SUITE(Conv1x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x3Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsSmallDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x3Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       ActivationFunctionsDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv1x3

TEST_SUITE(Conv5x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsSmallDataset ),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsDataset ),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv5x5

TEST_SUITE(Conv5x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv5x1

TEST_SUITE(Conv1x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv1x5
TEST_SUITE_END() // FP32


TEST_SUITE(FP16)

using CLWinogradConvolutionLayerFastMathFixture16 = WinogradConvolutionLayerFastMathValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, half, float>;
TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                               ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv3x3

TEST_SUITE(Conv3x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x1Dataset(),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsSmallDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x1Dataset(),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv3x1

TEST_SUITE(Conv1x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x3Dataset(),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsSmallDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x3Dataset(),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsDataset),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv1x3

TEST_SUITE(Conv5x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv5x5

TEST_SUITE(Conv5x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv5x1

TEST_SUITE(Conv1x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv1x5

TEST_SUITE_END() // FP16

TEST_SUITE_END() // ConvolutionLayer
TEST_SUITE_END() // Winograd
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
