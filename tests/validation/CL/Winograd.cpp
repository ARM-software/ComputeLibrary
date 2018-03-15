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
#include "tests/datasets/WinogradFilterTransformDataset.h"
#include "tests/datasets/WinogradInputTransformDataset.h"
#include "tests/datasets/WinogradOutputTransformDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WinogradLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f32(0.001f);
} // namespace

using namespace arm_compute::misc::shape_calculator;

TEST_SUITE(CL)
TEST_SUITE(Winograd)

TEST_SUITE(InputTransform)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
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
                                                framework::dataset::make("PadStrideInfo", {
                                                                                        PadStrideInfo(1, 1, 1, 0),
                                                                                        PadStrideInfo(1, 1, 0, 0),
                                                                                        PadStrideInfo(1, 1, 1, 1),
                                                                                        PadStrideInfo(2, 1, 1, 1),
                                                                                        PadStrideInfo(1, 1, 0, 1),
                                                                                        PadStrideInfo(1, 1, 0, 0),
                                                                                        PadStrideInfo(1, 1, 1, 1)
                                                                                    })),
                                                framework::dataset::make("KernelDims", {
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(5U, 5U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U)
                                                                                    })),
                                                framework::dataset::make("Expected", { false, false, false, false, false, false, false })),
                                            input_info, output_info, conv_info, kernel_dims, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradInputTransform::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, kernel_dims)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

using CLWinogradInputTransformFixture = WinogradInputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradInputTransform, float>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallWinogradInputTransformDataset(), datasets::LargeWinogradInputTransformDataset()),
                                                                   framework::dataset::make("DataType", { DataType::F32 })),
               shape_in, conv_info, kernel_dims, is_nchw_format, data_type)
{
    ARM_COMPUTE_UNUSED(is_nchw_format);

    TensorShape shape_out = compute_winograd_input_transform_shape(TensorInfo(shape_in, 1, data_type), conv_info, kernel_dims);

    // Create tensors
    CLTensor in  = create_tensor<CLTensor>(shape_in, data_type);
    CLTensor out = create_tensor<CLTensor>(shape_out, data_type);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradInputTransform winograd_input_transform;

    // Configure the function
    winograd_input_transform.configure(&in, &out, conv_info, kernel_dims);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallWinogradInputTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeWinogradInputTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // InputTransform

TEST_SUITE(FilterTransform)
// *INDENT-OFF*
// clang-format off
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
                                                framework::dataset::make("OutputTile", {
                                                                                        Size2D(2U, 2U),
                                                                                        Size2D(2U, 2U),
                                                                                        Size2D(2U, 2U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(2U, 2U),
                                                                                        Size2D(2U, 2U),
                                                                                        Size2D(4U, 4U)
                                                                                    })),
                                                framework::dataset::make("Expected", { false, false, false, false, true, true, true })),
                                            input_info, output_info, output_tile, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradFilterTransformKernel::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), output_tile)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

using CLWinogradFilterTransform        = CLSynthetizeFunctionWithZeroConstantBorder<CLWinogradFilterTransformKernel, 0>;
using CLWinogradFilterTransformFixture = WinogradFilterTransformValidationFixture<CLTensor, CLAccessor, CLWinogradFilterTransform, float>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallWinogradFilterTransformDataset(), datasets::LargeWinogradFilterTransformDataset()),
                                                                           framework::dataset::make("OutputTile", { Size2D(2U, 2U), Size2D(4U, 4U) })),
                                                                   framework::dataset::make("DataType", { DataType::F32 })),
               shape_a, is_nchw_format, output_tile, data_type)
{
    ARM_COMPUTE_UNUSED(is_nchw_format);

    TensorShape shape_b = compute_winograd_filter_transform_shape(TensorInfo(shape_a, 1, data_type), output_tile);

    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_a, data_type);
    CLTensor b = create_tensor<CLTensor>(shape_b, data_type);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradFilterTransform winograd_filter_transform;
    winograd_filter_transform.configure(&a, &b, output_tile);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixture, framework::DatasetMode::ALL, combine(combine(datasets::SmallWinogradFilterTransformDataset(), framework::dataset::make("OutputTile", { Size2D(2U, 2U), Size2D(4U, 4U) })),
                                                                                                        framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixture, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeWinogradFilterTransformDataset(),
                                                                                                                    framework::dataset::make("OutputTile", { Size2D(2U, 2U), Size2D(4U, 4U) })),
                                                                                                            framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // FilterTransform

TEST_SUITE(OutputTransform)
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(24U, 49U, 16U, 5U), 1, DataType::F16),        // F16 not supported
                                                                                        TensorInfo(TensorShape(128U, 3136U, 16U, 5U), 1, DataType::QASYMM8), // QASYMM8 not supported
                                                                                        TensorInfo(TensorShape(256U, 784U, 16U, 5U), 1, DataType::F32),      // Kernel size not supported
                                                                                        TensorInfo(TensorShape(512U, 169U, 16U, 5U), 1, DataType::F32),      // Valid
                                                                                        TensorInfo(TensorShape(13U, 6U, 16U, 4U), 1, DataType::F32),         // Padding needed
                                                                                        TensorInfo(TensorShape(7U, 16U, 16U, 7U), 1, DataType::F32),         // Valid
                                                                                        TensorInfo(TensorShape(1U, 442U, 16U, 37U), 1, DataType::F32)        // Wrong number of tiles
                                                                                    }),
                                                framework::dataset::make("BiasInfo", {
                                                                                        TensorInfo(TensorShape(24U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(128U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(256U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(512U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(13U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(1U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(14U, 14U, 24U, 5U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(112U, 112U, 128U, 5U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(55U, 55U, 256U, 5U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(26U, 26U, 512U, 5U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(5U, 4U, 13U, 4U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(8U, 8U, 7U, 7U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(51U, 33U, 1U, 37U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("KernelDims", {
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(5U, 5U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U)
                                                                                    })),
                                                framework::dataset::make("OutputDims", {
                                                                                        Size2D(14U, 14U),
                                                                                        Size2D(112U, 112U),
                                                                                        Size2D(55U, 55U),
                                                                                        Size2D(26U, 26U),
                                                                                        Size2D(5U, 4U),
                                                                                        Size2D(8U, 8U),
                                                                                        Size2D(51U, 33U)
                                                                                    })),
                                                framework::dataset::make("NumTiles", {
                                                                                        Size2D(7U, 7U),
                                                                                        Size2D(56U, 56U),
                                                                                        Size2D(28U, 28U),
                                                                                        Size2D(13U, 13U),
                                                                                        Size2D(3U, 2U),
                                                                                        Size2D(4U, 4U),
                                                                                        Size2D(26U, 16U)
                                                                                    })),
                                                framework::dataset::make("Expected", { false, false, false, true, false, true, false })),
                                            input_info, bias_info, output_info, kernel_dims, output_dims, num_tiles, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradOutputTransformKernel::validate(&input_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), kernel_dims, output_dims, num_tiles)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

using CLWinogradOutputTransform        = CLSynthetizeFunctionWithZeroConstantBorder<CLWinogradOutputTransformKernel, 0>;
using CLWinogradOutputTransformFixture = WinogradOutputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradOutputTransform, float>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallWinogradOutputTransformDataset(), datasets::LargeWinogradOutputTransformDataset()),
                                                                   framework::dataset::make("DataType", { DataType::F32 })),
               shape_a, kernel_dims, output_convolved_dims, num_tiles, data_layout, data_type)
{
    TensorShape shape_b = compute_winograd_output_transform_shape(TensorInfo(shape_a, 1, data_type), output_convolved_dims, data_layout);

    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_a, data_type);
    CLTensor b = create_tensor<CLTensor>(shape_b, data_type);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradOutputTransform winograd_output_transform;
    winograd_output_transform.configure(&a, nullptr, &b, kernel_dims, output_convolved_dims, num_tiles);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradOutputTransformFixture, framework::DatasetMode::ALL, combine(datasets::SmallWinogradOutputTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradOutputTransformFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeWinogradOutputTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // OutputTransform

TEST_SUITE(ConvolutionLayer)
// *INDENT-OFF*
// clang-format off
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
// clang-format on
// *INDENT-ON*

using CLWinogradConvolutionLayerFixture = WinogradConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLWinogradConvolutionLayer, float>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradConvolutionLayerFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                                                                                               framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradConvolutionLayerFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // ConvolutionLayer

TEST_SUITE_END() // Winograd
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
