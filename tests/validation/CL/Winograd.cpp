/*
 * Copyright (c) 2018 ARM Limited.
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
constexpr AbsoluteTolerance<float> tolerance_f32(0.001f);
constexpr AbsoluteTolerance<float> tolerance_convolution_layer_f32(0.1f);

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
                                                                               datasets::SmallWinogradInputTransformDataset4x4_5x5())));

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
                                      datasets::LargeWinogradInputTransformDataset4x4_5x5());

// Filter transform
const auto SmallWinogradFilterTransformDatasetNCHW =
           framework::dataset::concat(combine(datasets::Small3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 2U), Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Small3x1Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 1U), Size2D(4U, 1U) })),
           framework::dataset::concat(combine(datasets::Small1x3Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 2U), Size2D(1U, 4U) })),
           framework::dataset::concat(combine(datasets::Small5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Small5x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
                                      combine(datasets::Small1x5Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })))))));

const auto SmallWinogradFilterTransformDatasetNHWC =
           framework::dataset::concat(combine(datasets::Small3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Small3x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
           framework::dataset::concat(combine(datasets::Small1x3Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })),
                                      combine(datasets::Small5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })))));

const auto LargeWinogradFilterTransformDatasetNCHW =
           framework::dataset::concat(combine(datasets::Large3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 2U), Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Large3x1Shapes(), framework::dataset::make("OutputTile", { Size2D(2U, 1U), Size2D(4U, 1U) })),
           framework::dataset::concat(combine(datasets::Large1x3Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 2U), Size2D(1U, 4U) })),
           framework::dataset::concat(combine(datasets::Large5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
           framework::dataset::concat(combine(datasets::Large5x1Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 1U) })),
                                      combine(datasets::Large1x5Shapes(), framework::dataset::make("OutputTile", { Size2D(1U, 4U) })))))));

const auto LargeWinogradFilterTransformDatasetNHWC =
           framework::dataset::concat(combine(datasets::Large3x3Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })),
                                      combine(datasets::Large5x5Shapes(), framework::dataset::make("OutputTile", { Size2D(4U, 4U) })));

// Output transform
const auto SmallWinogradOutputTransformDatasetNCHW = datasets::SmallWinogradOutputTransformDatasetNCHW();

const auto SmallWinogradOutputTransformDatasetNHWC = datasets::SmallWinogradOutputTransformDatasetNHWC();

const auto LargeWinogradOutputTransformDatasetNCHW = datasets::LargeWinogradOutputTransformDatasetNCHW();

const auto LargeWinogradOutputTransformDatasetNHWC = datasets::LargeWinogradOutputTransformDatasetNHWC();
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

using CLWinogradInputTransformFixture = WinogradInputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradInputTransform, float>;

TEST_SUITE(NCHW)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(SmallWinogradInputTransformDatasetNCHW,
                                                                                                      LargeWinogradInputTransformDatasetNCHW),
                                                                           framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                           framework::dataset::make("DataType", { DataType::F32 })),
               shape_in, winograd_info, data_layout, data_type)
{
    TensorInfo  tensor_info_in(shape_in, 1, data_type);
    tensor_info_in.set_data_layout(data_layout);

    TensorShape shape_out = compute_winograd_input_transform_shape(tensor_info_in, winograd_info);

    // Create tensors
    CLTensor in  = create_tensor<CLTensor>(shape_in, data_type, 1, QuantizationInfo(), data_layout);
    CLTensor out = create_tensor<CLTensor>(shape_out, data_type);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradInputTransform winograd_input_transform;

    // Configure the function
    winograd_input_transform.configure(&in, &out, winograd_info);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixture, framework::DatasetMode::PRECOMMIT, combine(combine(SmallWinogradInputTransformDatasetNCHW,
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                     framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixture, framework::DatasetMode::NIGHTLY, combine(combine(LargeWinogradInputTransformDatasetNCHW,
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                   framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(SmallWinogradInputTransformDatasetNHWC,
                                                                                                      LargeWinogradInputTransformDatasetNHWC),
                                                                           framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                           framework::dataset::make("DataType", { DataType::F32 })),
               shape_in, winograd_info, data_layout, data_type)
{
    TensorShape shape_in_nhwc(shape_in);

    // Convert the shape to NHWC
    permute(shape_in_nhwc, PermutationVector(2U, 0U, 1U));

    // TensorInfo
    TensorInfo  tensor_info_in(shape_in_nhwc, 1, data_type);
    tensor_info_in.set_data_layout(data_layout);

    TensorShape shape_out = compute_winograd_input_transform_shape(tensor_info_in, winograd_info);

    // Create tensors
    CLTensor in  = create_tensor<CLTensor>(shape_in_nhwc, data_type, 1, QuantizationInfo(), data_layout);
    CLTensor out = create_tensor<CLTensor>(shape_out, data_type);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradInputTransform winograd_input_transform;

    // Configure the function
    winograd_input_transform.configure(&in, &out, winograd_info);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixture, framework::DatasetMode::PRECOMMIT, combine(combine(SmallWinogradInputTransformDatasetNHWC,
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                     framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixture, framework::DatasetMode::NIGHTLY, combine(combine(LargeWinogradInputTransformDatasetNHWC,
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // InputTransform

TEST_SUITE(FilterTransform)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 3U), 1, DataType::F16),     // F16 not supported
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
                                                framework::dataset::make("Expected", { false, false, false, false, true, true, true })),
                                            input_info, output_info, winograd_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradFilterTransformKernel::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), winograd_info)) == expected, framework::LogLevel::ERRORS);
}

using CLWinogradFilterTransform        = CLSynthetizeFunctionWithZeroConstantBorder<CLWinogradFilterTransformKernel, 0>;
using CLWinogradFilterTransformFixture = WinogradFilterTransformValidationFixture<CLTensor, CLAccessor, CLWinogradFilterTransform, float>;

TEST_SUITE(NCHW)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL,
               combine(combine(framework::dataset::concat(SmallWinogradFilterTransformDatasetNCHW,
                                                          LargeWinogradFilterTransformDatasetNCHW),
                                                          framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                          framework::dataset::make("DataType", { DataType::F32 })),
               shape_a, output_tile, data_layout, data_type)
{
    WinogradInfo winograd_info(output_tile, Size2D(shape_a[0], shape_a[1]), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, data_layout /* Not needed */);

    TensorShape shape_b = compute_winograd_filter_transform_shape(TensorInfo(shape_a, 1, data_type), winograd_info);

    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_a, data_type, 1, QuantizationInfo(), data_layout);
    CLTensor b = create_tensor<CLTensor>(shape_b, data_type, 1, QuantizationInfo(), data_layout);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradFilterTransform winograd_filter_transform;
    winograd_filter_transform.configure(&a, &b, winograd_info);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(SmallWinogradFilterTransformDatasetNCHW,
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradFilterTransformDatasetNCHW,
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL,
               combine(combine(framework::dataset::concat(SmallWinogradFilterTransformDatasetNHWC,
                                                          LargeWinogradFilterTransformDatasetNHWC),
                                                          framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                          framework::dataset::make("DataType", { DataType::F32 })),
               shape_in, output_tile, data_layout, data_type)
{
    TensorShape shape_in_nhwc(shape_in);

    // Convert the shape to NHWC
    permute(shape_in_nhwc, PermutationVector(2U, 0U, 1U));

    // TensorInfo
    TensorInfo  tensor_info_in(shape_in_nhwc, 1, data_type);
    tensor_info_in.set_data_layout(data_layout);

    WinogradInfo winograd_info(output_tile, Size2D(shape_in[0], shape_in[1]), Size2D() /* Not needed */, PadStrideInfo() /* Not needed */, data_layout /* Not needed */);

    TensorShape shape_b = compute_winograd_filter_transform_shape(tensor_info_in, winograd_info);

    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_in_nhwc, data_type, 1, QuantizationInfo(), data_layout);
    CLTensor b = create_tensor<CLTensor>(shape_b, data_type, 1, QuantizationInfo(), data_layout);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradFilterTransform winograd_filter_transform;
    winograd_filter_transform.configure(&a, &b, winograd_info);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(SmallWinogradFilterTransformDatasetNHWC,
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(LargeWinogradFilterTransformDatasetNHWC,
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                       framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // FilterTransform

TEST_SUITE(OutputTransform)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(512U, 49U, 16U, 5U), 1, DataType::F16),      // F16 not supported
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
                                                framework::dataset::make("Expected", { false, false, false, true, false, true, false, true, false })),
                                            input_info, bias_info, output_info, winograd_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradOutputTransformKernel::validate(&input_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), winograd_info)) == expected, framework::LogLevel::ERRORS);
}

using CLWinogradOutputTransform        = CLSynthetizeFunctionWithZeroConstantBorder<CLWinogradOutputTransformKernel, 0>;
using CLWinogradOutputTransformFixture = WinogradOutputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradOutputTransform, float>;

TEST_SUITE(NCHW)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(SmallWinogradOutputTransformDatasetNCHW,
                                                                                              LargeWinogradOutputTransformDatasetNCHW),
                                                                                              framework::dataset::make("DataType", { DataType::F32 })),
               shape_a, winograd_info, data_type)
{
    TensorShape shape_b = compute_winograd_output_transform_shape(TensorInfo(shape_a, 1, data_type), winograd_info);

    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_a, data_type);
    CLTensor b = create_tensor<CLTensor>(shape_b, data_type, 1, QuantizationInfo(), winograd_info.output_data_layout);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradOutputTransform winograd_output_transform;
    winograd_output_transform.configure(&a, nullptr, &b, winograd_info);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradOutputTransformFixture, framework::DatasetMode::ALL,
                       combine(SmallWinogradOutputTransformDatasetNCHW,
                               framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradOutputTransformFixture, framework::DatasetMode::NIGHTLY,
                       combine(LargeWinogradOutputTransformDatasetNCHW,
                               framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // NCHW

TEST_SUITE(NHWC)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(SmallWinogradOutputTransformDatasetNHWC,
                                                                                              LargeWinogradOutputTransformDatasetNHWC),
                                                                                              framework::dataset::make("DataType", { DataType::F32 })),
               shape_a, winograd_info, data_type)
{
    TensorShape shape_b = compute_winograd_output_transform_shape(TensorInfo(shape_a, 1, data_type), winograd_info);

    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_a, data_type);
    CLTensor b = create_tensor<CLTensor>(shape_b, data_type, 1, QuantizationInfo(), winograd_info.output_data_layout);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradOutputTransform winograd_output_transform;
    winograd_output_transform.configure(&a, nullptr, &b, winograd_info);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradOutputTransformFixture, framework::DatasetMode::ALL,
                       combine(SmallWinogradOutputTransformDatasetNHWC,
                               framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradOutputTransformFixture, framework::DatasetMode::NIGHTLY,
                       combine(LargeWinogradOutputTransformDatasetNHWC,
                               framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // OutputTransform

TEST_SUITE(ConvolutionLayer)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
                                                framework::dataset::make("InputInfo", {
                                                                                        TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F16),     // FP16 not supported
                                                                                        TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),     // Datatype mismatch
                                                                                        TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32), // Stride y not supported
                                                                                        TensorInfo(TensorShape(16U, 16U, 8U), 1, DataType::F32),     // Padding needed
                                                                                        TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32)  // Kernel size not supported
                                                                                      }),
                                                framework::dataset::make("WeightsInfo", {
                                                                                        TensorInfo(TensorShape(3U, 3U, 2U, 19U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(3U, 3U, 2U, 19U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(3U, 3U, 8U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16)
                                                                                        })),
                                                framework::dataset::make("BiasesInfo", {
                                                                                        TensorInfo(TensorShape(19U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(19U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(21U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(16U), 1, DataType::F32)
                                                                                       })),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F32),
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

using CLWinogradConvolutionLayerFastMathFixture = WinogradConvolutionLayerFastMathValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, float>;
TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv3x3

TEST_SUITE(Conv3x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer3x1Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer3x1Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv3x1

TEST_SUITE(Conv1x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x3Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x3Dataset(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                                       framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv1x3

TEST_SUITE(Conv5x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv5x5

TEST_SUITE(Conv5x1)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer5x1Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv5x1

TEST_SUITE(Conv1x5)
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFastMathFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer1x5Dataset(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))

{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_convolution_layer_f32);
}
TEST_SUITE_END() // Conv1x5

TEST_SUITE_END() // ConvolutionLayer
TEST_SUITE_END() // Winograd
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
