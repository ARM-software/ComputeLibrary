/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DeconvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f); /**< Tolerance for floating point tests */

const auto data3x3 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 2)
                     * framework::dataset::make("PadY", 0, 2) * framework::dataset::make("ax", 0) * framework::dataset::make("ay", 0) * framework::dataset::make("NumKernels", { 1, 3 });

const auto data1x1 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 1)
                     * framework::dataset::make("PadY", 0, 1) * framework::dataset::make("ax", 0) * framework::dataset::make("ay", 0) * framework::dataset::make("NumKernels", { 1, 3 });

} // namespace

TEST_SUITE(CL)
TEST_SUITE(DeconvolutionLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, (combine(datasets::SmallDeconvolutionShapes(), framework::dataset::make("DataType", DataType::F32))),
               input_shape, data_type)
{
    // Create shapes
    const unsigned int kernel_size_x = 3;
    const unsigned int kernel_size_y = 3;
    const unsigned int num_kernels   = 1;
    const TensorShape  weights_shape(kernel_size_x, kernel_size_y, input_shape.z(), num_kernels);
    const TensorShape  bias_shape(num_kernels);
    auto               out_dim      = deconvolution_output_dimensions(input_shape.x(), input_shape.y(), kernel_size_x, kernel_size_y, 1, 1, 0, 0, 1, 1);
    TensorShape        output_shape = deconvolution_output_shape(out_dim, input_shape, weights_shape);

    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(input_shape, data_type, 1);
    CLTensor weights = create_tensor<CLTensor>(weights_shape, data_type, 1);
    CLTensor bias    = create_tensor<CLTensor>(bias_shape, data_type, 1);
    CLTensor dst     = create_tensor<CLTensor>(output_shape, data_type, 1);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLDeconvolutionLayer deconv;
    deconv.configure(&src, &weights, &bias, &dst, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), 0, 0);

    // Validate valid region
    const ValidRegion src_valid_region     = shape_to_valid_region(input_shape);
    const ValidRegion weights_valid_region = shape_to_valid_region(weights_shape);
    const ValidRegion bias_valid_region    = shape_to_valid_region(bias_shape);
    const ValidRegion dst_valid_region     = shape_to_valid_region(output_shape);

    validate(src.info()->valid_region(), src_valid_region);
    validate(weights.info()->valid_region(), weights_valid_region);
    validate(bias.info()->valid_region(), bias_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),   // Mismatching data type
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),   // Invalid weights shape
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 4),   // Non supported data type
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 11),  // Invalid bias shape
                                            TensorInfo(TensorShape(13U, 11U, 4U, 3U), 1, DataType::F32, 0), // Window shrink
                                            TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32, 0),
                                          }),
    framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F16, 0),
                                            TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::QS8, 5),
                                            TensorInfo(TensorShape(3U, 2U, 2U, 2U), 1, DataType::F32, 11),
                                            TensorInfo(TensorShape(3U, 3U, 4U), 1, DataType::F32, 0),
                                              TensorInfo(TensorShape(1U, 1U, 2U, 4U), 1, DataType::F32, 0),
                                          })),
    framework::dataset::make("BiasInfo",  { TensorInfo(TensorShape(1U), 1, DataType::F16, 0),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32, 5),
                                            TensorInfo(TensorShape(25U, 11U), 1, DataType::F32, 11),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16, 0),
                                            TensorInfo(TensorShape(25U, 10U, 2U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32, 5),
                                            TensorInfo(TensorShape(13U, 13U, 2U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(11U, 9U, 1U, 3U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(32U, 16U, 4U), 1, DataType::F32, 0),
                                          })),
    framework::dataset::make("PadStrideInfo", { PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 1, 1),
                                                PadStrideInfo(1, 1, 0, 0),
                                           })),
    framework::dataset::make("ax",          {   1U,
                                                1U,
                                                1U,
                                                1U,
                                                0U,
                                                0U,
                                            })),
   framework::dataset::make("ay",           {   1U,
                                                1U,
                                                1U,
                                                1U,
                                                0U,
                                                0U,
                                            })),
    framework::dataset::make("Expected", { false, false, false, false, false, true })),
    input_info, weights_info, bias_info, output_info, pad_info, ax, ay, expected)
{
    bool is_valid = bool(CLDeconvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pad_info, ax, ay));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLDeconvolutionLayerFixture3x3 = DeconvolutionValidationFixture<CLTensor, CLAccessor, CLDeconvolutionLayer, T, 3, 3>;

template <typename T>
using CLDeconvolutionLayerFixture1x1 = DeconvolutionValidationFixture<CLTensor, CLAccessor, CLDeconvolutionLayer, T, 1, 1>;

TEST_SUITE(Float)

TEST_SUITE(FP32)
TEST_SUITE(W3x3)

FIXTURE_DATA_TEST_CASE(Run, CLDeconvolutionLayerFixture3x3<float>, framework::DatasetMode::ALL, combine(data3x3, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()

TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, CLDeconvolutionLayerFixture1x1<float>, framework::DatasetMode::ALL, combine(data1x1, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
