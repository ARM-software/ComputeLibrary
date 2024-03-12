/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLConv3D.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DirectConvolution3DFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const RelativeTolerance<half>        rel_tolerance_fp16(half(0.2)); /**< Relative tolerance for FP16 tests */
constexpr float                      abs_tolerance_fp16(0.05f);     /**< Absolute tolerance for FP16 tests */
constexpr RelativeTolerance<float>   rel_tolerance_fp32(0.05f);     /**< Relative tolerance for FP32 tests */
constexpr float                      abs_tolerance_fp32(0.0001f);   /**< Absolute tolerance for FP32 tests*/
constexpr AbsoluteTolerance<uint8_t> abs_tolerance_qasymm8(1);      /**< Absolute tolerance for quantized tests */
constexpr float                      tolerance_num = 0.07f;         /**< Tolerance number */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DirectConvolution3D)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 5U, 3U), // Unsupported data layout
                                                        TensorShape(27U, 13U, 5U, 3U), // Unsupported activation enabled
                                                        TensorShape(27U, 13U, 5U, 3U), // Mismatching data type
                                                        TensorShape(27U, 13U, 5U, 3U), // Unsupported data type
                                                        TensorShape(27U, 13U, 5U, 3U), // Mismatching input feature maps
                                                        TensorShape(27U, 13U, 5U, 3U), // Mismatching output feature maps
                                                        TensorShape(27U, 13U, 5U, 3U), // Mismatching bias shape
                                                        TensorShape(27U, 13U, 5U, 3U), // Unsupported number of weights dimensions
                                                        TensorShape(27U, 13U, 5U, 3U), // Unsupported number of biases dimensions
                                                        TensorShape(27U, 13U, 5U, 3U), // Mismatching output shape
                                                        TensorShape(27U, 13U, 5U, 3U)
                                                     }),
               framework::dataset::make("WeightsShape", { TensorShape(4U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 32U, 3U, 3U, 3U),
                                                          TensorShape(8U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U, 2U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U),
                                                          TensorShape(4U, 27U, 3U, 3U, 3U)
                                                     })),
               framework::dataset::make("BiasesShape", { TensorShape(4U),
                                                         TensorShape(4U),
                                                         TensorShape(4U),
                                                         TensorShape(4U),
                                                         TensorShape(4U),
                                                         TensorShape(4U),
                                                         TensorShape(8U),
                                                         TensorShape(4U),
                                                         TensorShape(4U),
                                                         TensorShape(4U),
                                                         TensorShape(4U)
                                                     })),
               framework::dataset::make("OutputShape", { TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U, 2U),
                                                         TensorShape(4U, 11U, 5U, 3U),
                                                         TensorShape(4U, 13U, 5U, 3U)
                                                     })),
               framework::dataset::make("Conv3dInfo",  { Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false),
                                                         Conv3dInfo(Size3D(1U, 1U, 1U), Padding3D(1U, 1U, 1U), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false)
                                                      })),
                framework::dataset::make("SrcDataType", { DataType::F32,
                                                          DataType::F32,
                                                          DataType::F32,
                                                          DataType::U32,
                                                          DataType::F32,
                                                          DataType::F32,
                                                          DataType::F32,
                                                          DataType::F32,
                                                          DataType::F32,
                                                          DataType::F32,
                                                          DataType::F32
                                                      })),
                framework::dataset::make("WeightsDataType", { DataType::F32,
                                                              DataType::F32,
                                                              DataType::F16,
                                                              DataType::U32,
                                                              DataType::F32,
                                                              DataType::F32,
                                                              DataType::F32,
                                                              DataType::F32,
                                                              DataType::F32,
                                                              DataType::F32,
                                                              DataType::F32
                                                      })),
                framework::dataset::make("DataLayout", { DataLayout::NCDHW,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC,
                                                         DataLayout::NDHWC
                                                      })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, false, false, true })),
               input_shape, weights_shape, biases_shape, output_shape, conv3d_info, src_data_type, weights_data_type, data_layout, expected)
{
    TensorInfo input_info   = TensorInfo(input_shape, 1, src_data_type);
    TensorInfo weights_info = TensorInfo(weights_shape, 1, weights_data_type);
    TensorInfo biases_info  = TensorInfo(biases_shape, 1, src_data_type);
    TensorInfo output_info  = TensorInfo(output_shape, 1, src_data_type);

    input_info.set_data_layout(data_layout);
    weights_info.set_data_layout(data_layout);
    biases_info.set_data_layout(data_layout);
    output_info.set_data_layout(data_layout);

    bool is_valid = bool(CLConv3D::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv3d_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using CLDirectConvolution3DFixture = DirectConvolution3DValidationFixture<CLTensor, CLAccessor, CLConv3D, T>;
template <typename T>
using CLDirectConvolution3DQuantizedFixture = DirectConvolution3DValidationQuantizedFixture<CLTensor, CLAccessor, CLConv3D, T>;

TEST_SUITE(NDHWC)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolution3DFixture<half>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(
                       framework::dataset::make("InputShape", { TensorShape(7U, 5U, 3U, 13U, 3U),
                                                                TensorShape(15U, 7U, 11U, 7U),
                                                                TensorShape(19U, 5U, 16U, 4U),
                                                                TensorShape(13U, 5U, 17U, 2U)
                                                              }),
                       framework::dataset::make("StrideX", { 1, 3, 2, 1 })),
                       framework::dataset::make("StrideY", { 2, 1, 3, 1 })),
                       framework::dataset::make("StrideZ", { 3, 2, 1, 1 })),
                       framework::dataset::make("PadX", { 0, 2, 1, 0 })),
                       framework::dataset::make("PadY", { 1, 0, 2, 0 })),
                       framework::dataset::make("PadZ", { 2, 1, 0, 0 })),
                       framework::dataset::make("KernelWidth", { 3, 7, 5, 1 })),
                       framework::dataset::make("KernelHeight", { 5, 3, 7, 1 })),
                       framework::dataset::make("KernelDepth", { 7, 5, 3, 1 })),
                       framework::dataset::make("NumKernels", { 5, 3, 1, 11 })),
                       framework::dataset::make("HasBias", { true, true, true, false })),
                       framework::dataset::make("Activation", ActivationLayerInfo())),
                       framework::dataset::make("DataType", DataType::F16)),
                       framework::dataset::make("DataLayout", DataLayout::NDHWC)))
{
    validate(CLAccessor(_target), _reference, rel_tolerance_fp16, tolerance_num, abs_tolerance_fp16);
}

TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolution3DFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(
                       framework::dataset::make("InputShape", { TensorShape(7U, 5U, 3U, 13U, 3U),
                                                                TensorShape(15U, 7U, 11U, 7U),
                                                                TensorShape(19U, 5U, 16U, 4U),
                                                                TensorShape(13U, 5U, 17U, 2U)
                                                              }),
                       framework::dataset::make("StrideX", { 1, 3, 2, 1 })),
                       framework::dataset::make("StrideY", { 2, 1, 3, 1 })),
                       framework::dataset::make("StrideZ", { 3, 2, 1, 1 })),
                       framework::dataset::make("PadX", { 0, 2, 1, 0 })),
                       framework::dataset::make("PadY", { 1, 0, 2, 0 })),
                       framework::dataset::make("PadZ", { 2, 1, 0, 0 })),
                       framework::dataset::make("KernelWidth", { 3, 7, 5, 1 })),
                       framework::dataset::make("KernelHeight", { 5, 3, 7, 1 })),
                       framework::dataset::make("KernelDepth", { 7, 5, 3, 1 })),
                       framework::dataset::make("NumKernels", { 5, 3, 1, 11 })),
                       framework::dataset::make("HasBias", { true, true, true, false })),
                       framework::dataset::make("Activation", ActivationLayerInfo())),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("DataLayout", DataLayout::NDHWC)))
{
    validate(CLAccessor(_target), _reference, rel_tolerance_fp32, 0.0, abs_tolerance_fp32);
}

// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // FP32

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolution3DQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(
                                                                                                                   framework::dataset::make("InputShape", { TensorShape(7U, 5U, 3U, 13U, 3U),
                                                                                                                           TensorShape(15U, 7U, 11U, 7U),
                                                                                                                           TensorShape(19U, 5U, 16U, 4U),
                                                                                                                           TensorShape(13U, 5U, 17U, 2U)
                                                                                                                                                          }),
                                                                                                                   framework::dataset::make("StrideX", { 1, 3, 2, 1 })),
                                                                                                               framework::dataset::make("StrideY", { 2, 1, 3, 1 })),
                                                                                                           framework::dataset::make("StrideZ", { 3, 2, 1, 1 })),
                                                                                                       framework::dataset::make("PadX", { 0, 2, 1, 0 })),
                                                                                                   framework::dataset::make("PadY", { 1, 0, 2, 0 })),
                                                                                               framework::dataset::make("PadZ", { 2, 1, 0, 0 })),
                                                                                           framework::dataset::make("KernelWidth", { 3, 7, 5, 1 })),
                                                                                       framework::dataset::make("KernelHeight", { 5, 3, 7, 1 })),
                                                                                   framework::dataset::make("KernelDepth", { 7, 5, 3, 1 })),
                                                                               framework::dataset::make("NumKernels", { 5, 3, 1, 11 })),
                                                                           framework::dataset::make("HasBias", { true, true, true, false })),
                                                                       framework::dataset::make("Activation", ActivationLayerInfo())),
                                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                                       framework::dataset::make("DataLayout", DataLayout::NDHWC)),
                                               framework::dataset::make("SrcQuantizationInfo", QuantizationInfo(0.1f, 10))),
                                       framework::dataset::make("WeightsQuantizationInfo", QuantizationInfo(0.3f, 20))),
                               framework::dataset::make("DstQuantizationInfo", QuantizationInfo(0.2f, 5))))
{
    validate(CLAccessor(_target), _reference, abs_tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolution3DQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(
                                                                                                                   framework::dataset::make("InputShape", { TensorShape(7U, 5U, 3U, 13U, 3U),
                                                                                                                           TensorShape(15U, 7U, 11U, 7U),
                                                                                                                           TensorShape(19U, 5U, 16U, 4U),
                                                                                                                           TensorShape(13U, 5U, 17U, 2U)
                                                                                                                                                          }),
                                                                                                                   framework::dataset::make("StrideX", { 1, 3, 2, 1 })),
                                                                                                               framework::dataset::make("StrideY", { 2, 1, 3, 1 })),
                                                                                                           framework::dataset::make("StrideZ", { 3, 2, 1, 1 })),
                                                                                                       framework::dataset::make("PadX", { 0, 2, 1, 0 })),
                                                                                                   framework::dataset::make("PadY", { 1, 0, 2, 0 })),
                                                                                               framework::dataset::make("PadZ", { 2, 1, 0, 0 })),
                                                                                           framework::dataset::make("KernelWidth", { 3, 7, 5, 1 })),
                                                                                       framework::dataset::make("KernelHeight", { 5, 3, 7, 1 })),
                                                                                   framework::dataset::make("KernelDepth", { 7, 5, 3, 1 })),
                                                                               framework::dataset::make("NumKernels", { 5, 3, 1, 11 })),
                                                                           framework::dataset::make("HasBias", { true, true, true, false })),
                                                                       framework::dataset::make("Activation", ActivationLayerInfo())),
                                                               framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                       framework::dataset::make("DataLayout", DataLayout::NDHWC)),
                                               framework::dataset::make("SrcQuantizationInfo", QuantizationInfo(0.1f, 10))),
                                       framework::dataset::make("WeightsQuantizationInfo", QuantizationInfo(0.3f, 20))),
                               framework::dataset::make("DstQuantizationInfo", QuantizationInfo(0.2f, 5))))
{
    validate(CLAccessor(_target), _reference, abs_tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // NDHWC
TEST_SUITE_END() // DirectConvolution3D
TEST_SUITE_END() // CL

} // namespace validation
} // namespace test
} // namespace arm_compute
