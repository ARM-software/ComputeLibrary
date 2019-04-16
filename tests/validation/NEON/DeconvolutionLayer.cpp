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
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
constexpr AbsoluteTolerance<float> tolerance_qasymm8(0.0); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
constexpr float                    tolerance_num = 0.07f;  /**< Tolerance number */

const auto data4x4 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 3)
                     * framework::dataset::make("PadY", 0, 3) * framework::dataset::make("NumKernels", { 3 });

const auto data3x3 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 2)
                     * framework::dataset::make("PadY", 0, 2) * framework::dataset::make("NumKernels", { 3 });

const auto data3x3_precommit = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 2) * framework::dataset::make("StrideY", 1, 2) * framework::dataset::make("PadX", 0, 2)
                               * framework::dataset::make("PadY", 0, 2) * framework::dataset::make("NumKernels", { 3 });

const auto data1x1 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 1)
                     * framework::dataset::make("PadY", 0, 1) * framework::dataset::make("NumKernels", { 3 });

const auto data_layouts_dataset = framework::dataset::make("DataLayout", { DataLayout::NCHW });
} // namespace

TEST_SUITE(NEON)
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
    auto               out_dim      = deconvolution_output_dimensions(input_shape.x(), input_shape.y(), kernel_size_x, kernel_size_y, 1, 1, 1, 1);
    TensorShape        output_shape = compute_deconvolution_output_shape(out_dim, TensorInfo(input_shape, 1, data_type), TensorInfo(weights_shape, 1, data_type));

    // Create tensors
    Tensor src     = create_tensor<Tensor>(input_shape, data_type, 1);
    Tensor weights = create_tensor<Tensor>(weights_shape, data_type, 1);
    Tensor bias    = create_tensor<Tensor>(bias_shape, data_type, 1);
    Tensor dst     = create_tensor<Tensor>(output_shape, data_type, 1);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEDeconvolutionLayer deconv;
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
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),   // Mismatching data type
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),   // Invalid weights shape
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),   // Non supported data type
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),  // Invalid bias shape
                                            TensorInfo(TensorShape(13U, 11U, 4U, 3U), 1, DataType::F32), // Window shrink
                                            TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),
                                          }),
    framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(3U, 2U, 2U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(3U, 3U, 4U), 1, DataType::F32),
                                              TensorInfo(TensorShape(1U, 1U, 2U, 4U), 1, DataType::F32),
                                          })),
    framework::dataset::make("BiasInfo",  { TensorInfo(TensorShape(1U), 1, DataType::F16),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(25U, 11U), 1, DataType::F32),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(4U), 1, DataType::F32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(25U, 10U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(13U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(11U, 9U, 1U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(32U, 16U, 4U), 1, DataType::F32),
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
    bool is_valid = bool(NEDeconvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pad_info, ax, ay));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEDeconvolutionLayerFixture4x4 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 4, 4>;

template <typename T>
using NEDeconvolutionLayerFixture3x3 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 3, 3>;

template <typename T>
using NEDeconvolutionLayerFixture1x1 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 1, 1>;

TEST_SUITE(Float)

TEST_SUITE(FP32)
TEST_SUITE(W4x4)

FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture4x4<float>, framework::DatasetMode::NIGHTLY, combine(combine(data4x4, framework::dataset::make("DataType", DataType::F32)),
                                                                                                            data_layouts_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // W4x4

TEST_SUITE(W3x3)

FIXTURE_DATA_TEST_CASE(RunSmall, NEDeconvolutionLayerFixture3x3<float>, framework::DatasetMode::PRECOMMIT, combine(combine(data3x3_precommit, framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   data_layouts_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDeconvolutionLayerFixture3x3<float>, framework::DatasetMode::NIGHTLY, combine(combine(data3x3, framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 data_layouts_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // W3x3

TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture1x1<float>, framework::DatasetMode::NIGHTLY, combine(combine(data1x1, framework::dataset::make("DataType", DataType::F32)),
                                                                                                            data_layouts_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // W1x1

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using NEDeconvolutionLayerQuantizedFixture4x4 = DeconvolutionValidationQuantizedFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 4, 4>;

template <typename T>
using NEDeconvolutionLayerQuantizedFixture3x3 = DeconvolutionValidationQuantizedFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 3, 3>;

template <typename T>
using NEDeconvolutionLayerQuantizedFixture1x1 = DeconvolutionValidationQuantizedFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 1, 1>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)

TEST_SUITE(W4x4)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerQuantizedFixture4x4<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data4x4, framework::dataset::make("DataType",
                                                                                                                       DataType::QASYMM8)),
                                                                                                                       data_layouts_dataset),
                                                                                                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255.f, 0))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8, tolerance_num);
}
TEST_SUITE_END() // W4x4

TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDeconvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data3x3_precommit, framework::dataset::make("DataType",
                       DataType::QASYMM8)),
                       data_layouts_dataset),
                       framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255.f, 0))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDeconvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data3x3, framework::dataset::make("DataType",
                       DataType::QASYMM8)),
                       data_layouts_dataset),
                       framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255.f, 0))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8, tolerance_num);
}
TEST_SUITE_END() // W3x3

TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerQuantizedFixture1x1<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data1x1, framework::dataset::make("DataType",
                                                                                                                       DataType::QASYMM8)),
                                                                                                                       data_layouts_dataset),
                                                                                                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255.f, 0))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8, tolerance_num);
}
TEST_SUITE_END() // W1x1

TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // DeconvolutionLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
