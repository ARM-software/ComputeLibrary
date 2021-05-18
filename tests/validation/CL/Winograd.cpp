/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"
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
const AbsoluteTolerance<half> tolerance_f16(half(1.f));
constexpr AbsoluteTolerance<float> tolerance_convolution_layer_f32(0.1f);
const AbsoluteTolerance<half> tolerance_convolution_layer_f16(half(0.4f));
RelativeTolerance<half_float::half> rel_tolerance_f16(half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for FP16 data types */
constexpr float                     tolerance_num   = 0.05f;  /**< Tolerance number */
constexpr float                     abs_tolerance_f32_nightly = 0.003f; /**< Absolute Tolerance value */
constexpr float                     abs_tolerance_convolution_layer_f16   = 2.5f;  /**< Tolerance number */
constexpr float                      tolerance_num_f16 = 0.15f;                 /**< Tolerance number */

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
using CLWinogradConvolutionLayerFastMathMixedDataLayoutFixture = WinogradConvolutionLayerFastMathValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, float, float, true, true>;
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
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLWinogradConvolutionLayerFastMathMixedDataLayoutFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(combine(
                                                framework::dataset::make("Input", TensorShape(8U, 8U, 32U)),
                                                framework::dataset::make("Weight", TensorShape(1U, 3U, 32U, 1U))),
                                                framework::dataset::make("Bias", TensorShape(1U))),
                                                framework::dataset::make("Output", TensorShape(8U, 6U, 1U))),
                                                framework::dataset::make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0))),
                                                framework::dataset::make("Dilation", Size2D(1U, 1U))),
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
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
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
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
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
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
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
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
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
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
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
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
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

TEST_SUITE(Conv1x7)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x7Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                       ActivationFunctionsSmallDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x7Dataset(),
                                               framework::dataset::make("DataType", { DataType::F16 })),
                                               ActivationFunctionsDataset),
                                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv1x7
TEST_SUITE_END() // FP16
TEST_SUITE_END() // ConvolutionLayer
TEST_SUITE_END() // Winograd
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
