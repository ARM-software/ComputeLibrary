/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DirectConvolutionLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DirectConvolutionLayerFixture.h"

/** Synced with tests/validation/dynamic_fusion/gpu/cl/DirectConv2d.cpp
 *  Please check there for any differences in the coverage
 */
namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<half>  tolerance_fp16(half(0.2));  /**< Tolerance for floating point tests */
RelativeTolerance<float> tolerance_fp32(0.05f);      /**< Tolerance for floating point tests */
constexpr float          abs_tolerance_f32(0.0001f); /**< Absolute tolerance for FP32 tests*/

constexpr float                      tolerance_num = 0.07f; /**< Tolerance number */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);  /**< Tolerance for quantized tests */

const auto data_strides          = combine(framework::dataset::make("StrideX", 1, 3), framework::dataset::make("StrideY", 1, 3));
const auto data_strides_small    = combine(framework::dataset::make("StrideX", 1), framework::dataset::make("StrideY", 1));
const auto data_ksize_one        = combine(framework::dataset::make("PadX", 0, 1), combine(framework::dataset::make("PadY", 0, 1), framework::dataset::make("KernelSize", 1)));
const auto data_ksize_one_small  = combine(framework::dataset::make("PadX", 0), combine(framework::dataset::make("PadY", 0), framework::dataset::make("KernelSize", 1)));
const auto data_ksize_three      = combine(framework::dataset::make("PadX", 0, 2), combine(framework::dataset::make("PadY", 0, 2), framework::dataset::make("KernelSize", 3)));
const auto data_ksize_five       = combine(framework::dataset::make("PadX", 0, 3), combine(framework::dataset::make("PadY", 0, 3), framework::dataset::make("KernelSize", 5)));
const auto data_ksize_nine       = combine(framework::dataset::make("PadX", 0, 3), combine(framework::dataset::make("PadY", 0, 3), framework::dataset::make("KernelSize", 9)));
const auto data_ksize_nine_small = combine(framework::dataset::make("PadX", 0, 1), combine(framework::dataset::make("PadY", 0, 1), framework::dataset::make("KernelSize", 9)));

const auto data_all_kernels = concat(concat(data_ksize_one, data_ksize_three), data_ksize_five);

const auto data          = combine(datasets::SmallDirectConvolutionShapes(), combine(data_strides, data_all_kernels));
const auto data9x9       = combine(datasets::SmallDirectConvolutionShapes(), combine(data_strides, data_ksize_nine));
const auto data_small    = combine(datasets::SmallDirectConvolutionShapes(), combine(data_strides_small, data_ksize_one_small));
const auto data_small9x9 = combine(datasets::SmallDirectConvolutionShapes(), combine(data_strides_small, data_ksize_nine_small));

/** Direct convolution nightly data set. */
const auto data_nightly         = combine(data, framework::dataset::make("NumKernels", { 1, 4 }));
const auto data_nightly_9x9     = combine(data9x9, framework::dataset::make("NumKernels", { 1, 4 }));
const auto data_nightly_usecase = combine(framework::dataset::make("InputShape", { TensorShape{ 3U, 800U, 800U } }),
                                          combine(framework::dataset::make("StrideX", { 1 }),
                                                  combine(framework::dataset::make("StrideY", { 1 }),
                                                          combine(framework::dataset::make("PadX", { 4 }),
                                                                  combine(framework::dataset::make("PadY", { 4 }),
                                                                          combine(framework::dataset::make("KernelSize", 9),
                                                                                  framework::dataset::make("NumKernels", { 16 })))))));

/** Direct convolution precommit data set. */
const auto data_precommit     = combine(data_small, framework::dataset::make("NumKernels", { 1 }));
const auto data_precommit_9x9 = combine(data_small9x9, framework::dataset::make("NumKernels", { 1 }));

/** Activation function Dataset*/
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f) });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DirectConvolutionLayer)

/** Check whether the configuration of a Direct Convolution layer with no
 * bias leads to a successful execution.
 */
TEST_CASE(NoBias, framework::DatasetMode::PRECOMMIT)
{
    const auto     src_shape     = TensorShape(27U, 13U, 2U);
    const auto     weights_shape = TensorShape(3U, 3U, 2U, 4U);
    const auto     bias_shape    = TensorShape(4U);
    const auto     dst_shape     = TensorShape(25U, 11U, 4U);
    constexpr auto dt            = DataType::F32;

    auto src     = create_tensor<CLTensor>(src_shape, dt);
    auto weights = create_tensor<CLTensor>(weights_shape, dt);
    auto dst     = create_tensor<CLTensor>(dst_shape, dt);

    const auto conv_info = PadStrideInfo(1, 1, 0, 0);

    // Create Direct Convolution function
    CLDirectConvolutionLayer conv{};
    conv.configure(&src, &weights, nullptr, &dst, conv_info);

    src.allocator()->allocate();
    weights.allocator()->allocate();
    dst.allocator()->allocate();

    library->fill_tensor_value(CLAccessor(src), 1.f);
    library->fill_tensor_value(CLAccessor(weights), 1.f);

    conv.run();

    // Compute reference to compare
    SimpleTensor<float> ref_src{ src_shape, dt };
    SimpleTensor<float> ref_weights{ weights_shape, dt };
    SimpleTensor<float> ref_bias{ bias_shape, dt };
    library->fill_tensor_value(ref_src, 1.f);
    library->fill_tensor_value(ref_weights, 1.f);
    // No bias
    library->fill_tensor_value(ref_bias, 0.f);
    auto ref_dst = reference::convolution_layer<float>(ref_src, ref_weights, ref_bias, dst_shape, conv_info);

    validate(CLAccessor(dst), ref_dst);
}

/** Check whether the case of rectangle kernels i.e. when width and height of the weight_shape are not equal
 *  would lead to successful run
 */
TEST_CASE(NonSquareKernel, framework::DatasetMode::PRECOMMIT)
{
    auto           src_shape     = TensorShape(33U, 27U, 3U);
    auto           weights_shape = TensorShape(5U, 7U, 3U, 4U); // non-square kernel
    const auto     bias_shape    = TensorShape(4U);
    auto           dst_shape     = TensorShape(11U, 12U, 4U);
    constexpr auto dt            = DataType::F32;

    TensorShape src_shape_nhwc(src_shape);
    TensorShape weights_shape_nhwc(weights_shape);
    TensorShape dst_shape_nhwc(dst_shape);

    // Non-square shapes are only allowed for NHWC
    permute(src_shape_nhwc, PermutationVector(2U, 0U, 1U));
    permute(weights_shape_nhwc, PermutationVector(2U, 0U, 1U));
    permute(dst_shape_nhwc, PermutationVector(2U, 0U, 1U));

    auto       src       = create_tensor<CLTensor>(src_shape_nhwc, dt, 1, QuantizationInfo(), DataLayout::NHWC);
    auto       weights   = create_tensor<CLTensor>(weights_shape_nhwc, dt, 1, QuantizationInfo(), DataLayout::NHWC);
    auto       dst       = create_tensor<CLTensor>(dst_shape_nhwc, dt, 1, QuantizationInfo(), DataLayout::NHWC);
    const auto conv_info = PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR);

    // Create direct convolution function
    CLDirectConvolutionLayer conv{};
    conv.configure(&src, &weights, nullptr, &dst, conv_info);

    src.allocator()->allocate();
    weights.allocator()->allocate();
    dst.allocator()->allocate();

    library->fill_tensor_value(CLAccessor(src), 1.f);
    library->fill_tensor_value(CLAccessor(weights), 1.f);

    conv.run();

    // Compute reference to compare
    SimpleTensor<float> ref_src{ src_shape, dt };
    SimpleTensor<float> ref_weights{ weights_shape, dt };
    SimpleTensor<float> ref_bias{ bias_shape, dt };
    library->fill_tensor_value(ref_src, 1.f);
    library->fill_tensor_value(ref_weights, 1.f);
    // No bias
    library->fill_tensor_value(ref_bias, 0.f);
    auto ref_dst = reference::convolution_layer<float>(ref_src, ref_weights, ref_bias, dst_shape, conv_info);

    validate(CLAccessor(dst), ref_dst);
}
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid: Mismatching data type input/weights
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid: Mismatching input feature maps
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid weights dimensions
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Unsupported biases size
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Unsupported biases dimensions
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid output size
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),
                                                     }),
               framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(3U, 3U, 3U, 4U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U, 3U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(1U, 1U, 2U, 4U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("BiasesInfo",{ TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(26U, 11U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 16U, 4U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("ConvInfo",  { PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                      })),
                       framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)
})),
               framework::dataset::make("Expected", { false, false, false, false, false, false, true })),
               input_info, weights_info, biases_info, output_info, conv_info, act_info, expected)
{
    bool is_valid = bool(CLDirectConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, act_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLDirectConvolutionLayerFixture = DirectConvolutionValidationFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;
template <typename T>
using CLDirectConvolutionLayerMixedDataLayoutFixture = DirectConvolutionValidationFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T, true>;
template <typename T>
using CLDirectConvolutionValidationWithTensorShapesFixture = DirectConvolutionValidationWithTensorShapesFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;
template <typename T>
using CLDirectConvolutionLayerQuantizedFixture = DirectConvolutionValidationQuantizedFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;
template <typename T>
using CLDirectConvolutionLayerQuantizedMixedDataLayoutFixture = DirectConvolutionValidationQuantizedFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T, true>;
template <typename T>
using CLDirectConvolutionValidationWithTensorShapesQuantizedFixture = DirectConvolutionValidationWithTensorShapesQuantizedFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;

TEST_SUITE(NHWC)
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", {
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC), // Arbitrary weight sizes for NHWC are supported
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC), // Non-rectangular weights dimensions for NHWC are supported
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC), // Strides > 2 for any kernel sizes for NHWC are supported
                                                     }),
               framework::dataset::make("WeightsInfo",{
                                                        TensorInfo(TensorShape(2U, 13U, 13U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 5U, 3U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 3U, 3U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                     })),
               framework::dataset::make("BiasesInfo",{
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                     })),
               framework::dataset::make("OutputInfo",{
                                                       TensorInfo(TensorShape(4U, 15U, 1U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U, 23U, 11U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U, 9U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                     })),
               framework::dataset::make("ConvInfo",  {
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(3, 3, 0, 0),
                                                      })),
                       framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
})),
               framework::dataset::make("Expected", { true, true, true })),
               input_info, weights_info, biases_info, output_info, conv_info, act_info, expected)
{
    bool is_valid = bool(CLDirectConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, act_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1, 3, 1, 1 })),
               framework::dataset::make("StrideY", { 1, 3, 2, 1 })),
               framework::dataset::make("PadX", { 1, 3, 0, 4 })),
               framework::dataset::make("PadY", { 1, 3, 0, 4 })),
               framework::dataset::make("KernelSize", { 3, 8, 1, 9 })),
               framework::dataset::make("NumKernels", { 17, 3, 1, 19 })),
               framework::dataset::make("DataType",  DataType::F16)),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_fp16, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLDirectConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
               combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(800U, 800U, 3U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 1 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 9 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::F16)),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::IDENTITY) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_fp16, tolerance_num);
}

TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1, 3, 1, 1 })),
               framework::dataset::make("StrideY", { 1, 3, 2, 1 })),
               framework::dataset::make("PadX", { 1, 3, 0, 4 })),
               framework::dataset::make("PadY", { 1, 3, 0, 4 })),
               framework::dataset::make("KernelSize", { 3, 8, 1, 9 })),
               framework::dataset::make("NumKernels", { 17, 3, 1, 19 })),
               framework::dataset::make("DataType",  DataType::F32)),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLDirectConvolutionLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 2 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 3 })),
               framework::dataset::make("KernelSize", { 3 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::F32)),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDirectConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
               combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(800U, 800U, 3U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 1 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 9 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::F32)),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::IDENTITY) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.0, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1, 3, 1, 1 })),
               framework::dataset::make("StrideY", { 1, 3, 2, 1 })),
               framework::dataset::make("PadX", { 1, 3, 0, 4 })),
               framework::dataset::make("PadY", { 1, 3, 0, 4 })),
               framework::dataset::make("KernelSize", { 3, 8, 1, 9 })),
               framework::dataset::make("NumKernels", { 7, 3, 1, 3 })),
               framework::dataset::make("DataType",  DataType::QASYMM8)),
               framework::dataset::make("QuantizationInfo", QuantizationInfo(1.1f / 255, 10))),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLDirectConvolutionLayerQuantizedMixedDataLayoutFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 2 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 3 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::QASYMM8)),
               framework::dataset::make("QuantizationInfo", QuantizationInfo(1.1f / 255, 10))),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDirectConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
               combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(800U, 800U, 3U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 1 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 9 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::QASYMM8)),
               framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255, 10))),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1, 3, 1, 1 })),
               framework::dataset::make("StrideY", { 1, 3, 2, 1 })),
               framework::dataset::make("PadX", { 1, 3, 0, 4 })),
               framework::dataset::make("PadY", { 1, 3, 0, 4 })),
               framework::dataset::make("KernelSize", { 3, 8, 1, 9 })),
               framework::dataset::make("NumKernels", { 7, 3, 1, 3 })),
               framework::dataset::make("DataType",  DataType::QASYMM8_SIGNED)),
               framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255, 10))),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLDirectConvolutionLayerQuantizedMixedDataLayoutFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 1 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 3 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::QASYMM8_SIGNED)),
               framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255, 10))),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDirectConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY,
               combine(combine(combine(combine(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(800U, 800U, 3U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 1 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 9 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::QASYMM8_SIGNED)),
               framework::dataset::make("QuantizationInfo", QuantizationInfo(2.f / 255, 10))),
               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) )),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // NHWC

TEST_SUITE(NCHW)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", {
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, DataLayout::NCHW), // Unsupported kernel width
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, DataLayout::NCHW), // Non-rectangular weights dimensions are unsupported
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, DataLayout::NCHW)  // Unsupported stride
                                                     }),
               framework::dataset::make("WeightsInfo",{
                                                        TensorInfo(TensorShape(11U, 11U, 2U, 4U), 1, DataType::F32, DataLayout::NCHW),
                                                        TensorInfo(TensorShape(5U, 3U, 2U, 4U), 1, DataType::F32, DataLayout::NCHW),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, DataLayout::NCHW)
                                                     })),
               framework::dataset::make("BiasesInfo",{
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NCHW),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NCHW),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NCHW)
                                                     })),
               framework::dataset::make("OutputInfo",{
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, DataLayout::NCHW),
                                                       TensorInfo(TensorShape(23U, 11U, 4U), 1, DataType::F32, DataLayout::NCHW),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, DataLayout::NCHW)
                                                     })),
               framework::dataset::make("ConvInfo",  {
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(3, 3, 0, 0)
                                                      })),
                       framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)
})),
               framework::dataset::make("Expected", { false, false, false})),
               input_info, weights_info, biases_info, output_info, conv_info, act_info, expected)
{
    bool is_valid = bool(CLDirectConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, act_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data_precommit, framework::dataset::make("DataType", DataType::F16)),
                                                                                                                   ActivationFunctionsDataset),
                                                                                                                   framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDirectConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data_nightly, framework::dataset::make("DataType", DataType::F16)),
                                                                                                                 ActivationFunctionsDataset),
                                                                                                                 framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, tolerance_num);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data_precommit, framework::dataset::make("DataType",
                                                                                                                    DataType::F32)),
                                                                                                                    ActivationFunctionsDataset),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLDirectConvolutionLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data_precommit,
                       framework::dataset::make("DataType",
                                                DataType::F32)),
                       ActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDirectConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data_nightly, framework::dataset::make("DataType", DataType::F32)),
                                                                                                                  ActivationFunctionsDataset),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.0, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP32_CustomDataset)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionValidationWithTensorShapesFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::DirectConvolutionLayerDataset(),
                       framework::dataset::make("DataType", DataType::F32)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.0, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32_CustomDataset
TEST_SUITE_END() // Float

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLDirectConvolutionLayerQuantizedMixedDataLayoutFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(data_precommit,
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 10) })),
                       QuantizedActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(data_precommit,
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 10), QuantizationInfo(1.1f, 10) })),
                       QuantizedActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmall9x9, CLDirectConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(data_precommit_9x9,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(3.f / 255, 10), QuantizationInfo(1.1f, 10) })),
                       QuantizedActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDirectConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(data_nightly, framework::dataset::make("DataType",
                       DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 10), QuantizationInfo(1.1f, 10) })),
                       QuantizedActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge9x9, CLDirectConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(data_nightly_9x9,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(3.f / 255, 10), QuantizationInfo(1.1f, 10) })),
                       QuantizedActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_CustomDataset)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionValidationWithTensorShapesQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::DirectConvolutionLayerDataset(),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 127), QuantizationInfo(1.1f, 10) })),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_CustomDataset

TEST_SUITE(QASYMM8_SIGNED)

FIXTURE_DATA_TEST_CASE(RunSmall, CLDirectConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(data_precommit, framework::dataset::make("DataType",
                                                                                                                        DataType::QASYMM8_SIGNED)),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 10), QuantizationInfo(1.1f, -10) })),
                                                                                                                        QuantizedActivationFunctionsDataset),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLDirectConvolutionLayerQuantizedMixedDataLayoutFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(data_precommit,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8_SIGNED)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.1f, -10) })),
                       QuantizedActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmall9x9, CLDirectConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(data_precommit_9x9,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8_SIGNED)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 10), QuantizationInfo(1.1f, 10) })),
                       QuantizedActivationFunctionsDataset),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunCustomDataset, CLDirectConvolutionValidationWithTensorShapesQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::DirectConvolutionLayerDataset(),
                                                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                               framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 127), QuantizationInfo(1.1f, 10) })),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // NCHW
TEST_SUITE_END() // DirectConvolutionLayer
TEST_SUITE_END() // CL

} // namespace validation
} // namespace test
} // namespace arm_compute
