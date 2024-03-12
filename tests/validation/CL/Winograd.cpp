/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "tests/datasets/ActivationFunctionsDataset.h"
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
using framework::dataset::make;
namespace
{
// *INDENT-OFF*
// clang-format off
const AbsoluteTolerance<half> tolerance_f16(half(1.f));
constexpr AbsoluteTolerance<float> tolerance_convolution_layer_f32(0.1f);
const AbsoluteTolerance<half> tolerance_convolution_layer_f16(half(0.4f));
RelativeTolerance<half_float::half> rel_tolerance_f16(half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for FP16 data types */
constexpr float                     tolerance_num   = 0.05f;  /**< Tolerance number */
constexpr float                     abs_tolerance_convolution_layer_f16   = 2.5f;  /**< Tolerance number */
constexpr float                     tolerance_num_f16 = 0.15f;                 /**< Tolerance number */

const auto ActivationFunctionsDataset = make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.8f),
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

const auto ActivationFunctionsSmallDataset = make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.8f, -0.5f)
});

} // namespace

using namespace arm_compute::misc::shape_calculator;

/*
    Testing Strategy of CL Winograd:
        - For nchw and nhwc and for each kernel size, we have a dedicated OpenCL kernel.
          (except 1xN and Nx1 uses NxN under the hood). Therefore, test cases should be
          stressed for each of these configurations.
        - Fp32 and Fp16 kernels are the same. Only the DATA_TYPE build option changes
          between these two. Because the same kernel is stressed thoroughly for both
          small and large shapes for Fp32 data type, Fp16 kernels are run on a subset
          of the shapes, because we get diminishing returns by exhaustively testing the
          same kernel.
        - Activations only affect the output stage and it's calculated on the output tile.
          Exhaustively testing all activations with all the shapes does not provide much
          value but increases the testing time quite significantly. Therefore, all activations
          are tested in a subset of the shapes, and for all MxM kernels and data layouts as
          they represent different OpenCL kernels. (1xM and Mx1 kernels use MxM under the hood).
*/
TEST_SUITE(CL)
TEST_SUITE(Winograd)

TEST_SUITE(ConvolutionLayer)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", {
        TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F16),     // Insufficient padding
        TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),     // Datatype mismatch
        TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32), // Stride y not supported
        TensorInfo(TensorShape(16U, 16U, 8U), 1, DataType::F32),     // Padding needed
        TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32)  // Kernel size not supported
        }),
    make("WeightsInfo", {
        TensorInfo(TensorShape(3U, 3U, 2U, 19U), 1, DataType::F16),
        TensorInfo(TensorShape(3U, 3U, 2U, 19U), 1, DataType::QASYMM8),
        TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
        TensorInfo(TensorShape(3U, 3U, 8U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16)
        }),
    make("BiasesInfo", {
        TensorInfo(TensorShape(19U), 1, DataType::F16),
        TensorInfo(TensorShape(19U), 1, DataType::F32),
        TensorInfo(TensorShape(21U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32)
        }),
    make("OutputInfo", {
        TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F16),
        TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
        TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 16U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32)
        }),
    make("ConvInfo", {
        PadStrideInfo(1, 1, 1, 1),
        PadStrideInfo(1, 1, 1, 1),
        PadStrideInfo(1, 2, 0, 0),
        PadStrideInfo(1, 1, 1, 1),
        PadStrideInfo(1, 1, 1, 0)
    }),
    make("Expected", { false, false, false, false, false })),
    input_info, weights_info, bias_info, output_info, conv_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info)) == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(SupportedKernels, framework::DatasetMode::ALL, zip(
    make("WeightsInfo", {
        // Shapes are always in NCHW format. When layout is NHWC, the shape is permuted

        // Fp32/16, NCHW
        // 3x1, 1x3, 3x3 --> all TRUE
        TensorInfo(TensorShape(3U, 3U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
        TensorInfo(TensorShape(1U, 3U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
        TensorInfo(TensorShape(3U, 1U, 2U, 8U), 1, DataType::F16, DataLayout::NCHW),

        // 5x1, 1x5, 5x5 --> all TRUE
        TensorInfo(TensorShape(5U, 5U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
        TensorInfo(TensorShape(1U, 5U, 2U, 8U), 1, DataType::F16, DataLayout::NCHW),
        TensorInfo(TensorShape(5U, 1U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),

        // 7x1, 1x7, 7x7
        // nchw does not support kernels with size 7 --> all FALSE
        TensorInfo(TensorShape(7U, 7U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
        TensorInfo(TensorShape(1U, 7U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
        TensorInfo(TensorShape(7U, 1U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),

        // unsupported kernel sizes
        TensorInfo(TensorShape(2U, 2U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
        TensorInfo(TensorShape(5U, 2U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
        TensorInfo(TensorShape(3U, 6U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),

        // Fp32/16, NHWC
        // 7x1, 1x7, 7x7 --> all TRUE
        TensorInfo(TensorShape(7U, 7U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
        TensorInfo(TensorShape(1U, 7U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
        TensorInfo(TensorShape(7U, 1U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),

        // 3x1, 1x3, 3x3 --> all TRUE
        TensorInfo(TensorShape(3U, 3U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
        TensorInfo(TensorShape(1U, 3U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
        TensorInfo(TensorShape(3U, 1U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),

        // 5x1, 1x5, 5x5 --> all TRUE
        TensorInfo(TensorShape(5U, 5U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
        TensorInfo(TensorShape(1U, 5U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
        TensorInfo(TensorShape(5U, 1U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),

        // unsupported kernel sizes
        TensorInfo(TensorShape(2U, 2U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
        TensorInfo(TensorShape(5U, 2U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
        TensorInfo(TensorShape(3U, 6U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),

        }),
    make("Expected", {
        true, true, true,     // nchw, 3x3, 1x3, 3x1
        true, true, true,     // nchw, 5x5, 1x5, 5x1
        false, false, false,  // nchw, 7x7, 1x7, 7x1
        false, false, false,  // nchw, random unsupported kernels
        true, true, true,     // nhwc, 7x7, 1x7, 7x1
        true, true, true,     // nhwc, 3x3, 1x3, 3x1
        true, true, true,     // nhwc, 5x5, 1x5, 5x1
        false, false, false,  // nchw, random unsupported kernels
    })),
    weights_info_const, expected)
{
    DataType data_type = weights_info_const.data_type();
    DataLayout data_layout = weights_info_const.data_layout();

    TensorInfo input_info = TensorInfo(TensorShape(17U, 31U, 2U), 1, data_type);
    TensorInfo bias_info = TensorInfo(TensorShape(8U), 1, data_type);
    TensorInfo weights_info = weights_info_const;

    if(data_layout == DataLayout::NHWC)
    {
        // Convert to NHWC
        PermutationVector perm = PermutationVector(2U, 0U, 1U);

        TensorShape input_shape = input_info.tensor_shape();
        TensorShape weights_shape = weights_info.tensor_shape();
        permute(input_shape, perm);
        permute(weights_shape, perm);

        input_info.set_tensor_shape(input_shape);
        weights_info.set_tensor_shape(weights_shape);

        input_info.set_data_layout(data_layout);
        weights_info.set_data_layout(data_layout);
        bias_info.set_data_layout(data_layout);
    }

    PadStrideInfo conv_info(1, 1, 0, 0);

    TensorShape output_shape = compute_deep_convolution_shape(input_info, weights_info, conv_info);
    TensorInfo output_info = TensorInfo(output_shape, 1, data_type, data_layout);

    Status status = CLWinogradConvolutionLayer::validate(
        &input_info,
        &weights_info,
        &bias_info,
        &output_info,
        conv_info,
        ActivationLayerInfo(),
        true /* fast math */);

    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE(FP32)
using CLWinogradConvolutionLayerFastMathFixture = WinogradConvolutionLayerFastMathValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, float>;
using CLWinogradConvolutionLayerFastMathMixedDataLayoutFixture = WinogradConvolutionLayerFastMathValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, float, float, true, true>;
TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                               make("DataType", { DataType::F32 }),
                                               ActivationFunctionsSmallDataset,
                                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                            make("DataType", { DataType::F32 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunActivations, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(
                            make("Input", TensorShape(8U, 8U, 32U)),
                            make("Weight", TensorShape(3U, 3U, 32U, 4U)),
                            make("Bias", TensorShape(4U)),
                            make("Output", TensorShape(6U, 6U, 4U)),
                            make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                            make("Dilation", Size2D(1U, 1U)),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv3x3

TEST_SUITE(Conv3x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x1Dataset(),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer3x1Dataset(),
                            make("DataType", { DataType::F32 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv3x1

TEST_SUITE(Conv1x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x3Dataset(),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLWinogradConvolutionLayerFastMathMixedDataLayoutFixture, framework::DatasetMode::PRECOMMIT,
                       combine(
                            make("Input", TensorShape(8U, 8U, 32U)),
                            make("Weight", TensorShape(1U, 3U, 32U, 1U)),
                            make("Bias", TensorShape(1U)),
                            make("Output", TensorShape(8U, 6U, 1U)),
                            make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                            make("Dilation", Size2D(1U, 1U)),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer1x3Dataset(),
                            make("DataType", { DataType::F32 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv1x3

TEST_SUITE(Conv5x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer5x5Dataset(),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer5x5Dataset(),
                            make("DataType", { DataType::F32 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunActivations, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(
                            make("Input", TensorShape(13U, 13U, 32U)),
                            make("Weight", TensorShape(5U, 5U, 32U, 4U)),
                            make("Bias", TensorShape(4U)),
                            make("Output", TensorShape(9U, 9U, 4U)),
                            make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                            make("Dilation", Size2D(1U, 1U)),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv5x5

TEST_SUITE(Conv5x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer5x1Dataset(),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer5x1Dataset(),
                            make("DataType", { DataType::F32 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv5x1

TEST_SUITE(Conv1x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x5Dataset(),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer1x5Dataset(),
                            make("DataType", { DataType::F32 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv1x5

TEST_SUITE(Conv1x7)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x7Dataset(),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunActivations, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(
                            make("Input", TensorShape(13U, 13U, 32U)),
                            make("Weight", TensorShape(1U, 7U, 32U, 4U)),
                            make("Bias", TensorShape(4U)),
                            make("Output", TensorShape(13U, 11U, 4U)),
                            make("PadStrideInfo", PadStrideInfo(1, 1, 0, 2)),
                            make("Dilation", Size2D(1U, 1U)),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsDataset,
                            make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv1x7

TEST_SUITE(Conv7x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer7x1Dataset(),
                            make("DataType", { DataType::F32 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv7x1

/** @note: Although 7x7 is in the kernels, reference implementation
 *  does not support it. So, it remains as a "test gap".
 */

TEST_SUITE_END() // FP32


TEST_SUITE(FP16)

using CLWinogradConvolutionLayerFastMathFixture16 = WinogradConvolutionLayerFastMathValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, half, float>;
TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer3x3DatasetFp16Subset(),
                            make("DataType", { DataType::F16 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunActivations, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(
                            make("Input", TensorShape(8U, 8U, 32U)),
                            make("Weight", TensorShape(3U, 3U, 32U, 6U)),
                            make("Bias", TensorShape(6U)),
                            make("Output", TensorShape(6U, 6U, 6U)),
                            make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                            make("Dilation", Size2D(1U, 1U)),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv3x3

TEST_SUITE(Conv3x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x1Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer3x1DatasetFp16Subset(),
                            make("DataType", { DataType::F16 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv3x1

TEST_SUITE(Conv1x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x3Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer1x3DatasetFp16Subset(),
                            make("DataType", { DataType::F16 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv1x3

TEST_SUITE(Conv5x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer5x5Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer5x5DatasetFp16Subset(),
                            make("DataType", { DataType::F16 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunActivations, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(
                            make("Input", TensorShape(13U, 13U, 32U)),
                            make("Weight", TensorShape(5U, 5U, 32U, 6U)),
                            make("Bias", TensorShape(6U)),
                            make("Output", TensorShape(9U, 9U, 6U)),
                            make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                            make("Dilation", Size2D(1U, 1U)),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv5x5

TEST_SUITE(Conv5x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer5x1Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer5x1DatasetFp16Subset(),
                            make("DataType", { DataType::F16 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv5x1

TEST_SUITE(Conv1x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x5Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer1x5DatasetFp16Subset(),
                            make("DataType", { DataType::F16 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv1x5

TEST_SUITE(Conv1x7)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x7Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer1x7DatasetFp16Subset(),
                            make("DataType", { DataType::F16 }),
                            make("ActivationInfo", { ActivationLayerInfo() }),
                            make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}

FIXTURE_DATA_TEST_CASE(RunActivations, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(
                            make("Input", TensorShape(13U, 13U, 32U)),
                            make("Weight", TensorShape(1U, 7U, 32U, 6U)),
                            make("Bias", TensorShape(6U)),
                            make("Output", TensorShape(13U, 7U, 6U)),
                            make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                            make("Dilation", Size2D(1U, 1U)),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsDataset,
                            make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_convolution_layer_f16);
}
TEST_SUITE_END() // Conv1x7

TEST_SUITE(Conv7x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer7x1Dataset(),
                            make("DataType", { DataType::F16 }),
                            ActivationFunctionsSmallDataset,
                            make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
}
TEST_SUITE_END() // Conv7x1

TEST_SUITE_END() // FP16
TEST_SUITE_END() // ConvolutionLayer
TEST_SUITE_END() // Winograd
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
