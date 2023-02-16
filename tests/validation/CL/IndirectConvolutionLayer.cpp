/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLIndirectConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DirectConvolutionLayerFixture.h"

// Note: Since the interface of indirect convolution is the same of direct convolution, we can reuse
// the direct convolution fixture

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
constexpr float          tolerance_num = 0.07f;      /**< Tolerance number */

/** Activation function Dataset*/
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f) });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(IndirectConvolutionLayer)

/** Check whether the configuration of a indirect convolution layer with no
 * bias leads to a successful run.
 */
TEST_CASE(NoBias, framework::DatasetMode::PRECOMMIT)
{
    const TensorShape    src_shape_nhwc = TensorShape(8U, 27U, 13U);
    const TensorShape    wei_shape_nhwc = TensorShape(8U, 3U, 3U, 4U);
    const TensorShape    bia_shape      = TensorShape(4U);
    const TensorShape    dst_shape_nhwc = TensorShape(4U, 25U, 11U);
    constexpr DataType   dt             = DataType::F32;
    constexpr DataLayout data_layout    = DataLayout::NHWC;

    auto src_nhwc = create_tensor<CLTensor>(src_shape_nhwc, dt, 1, QuantizationInfo(), data_layout);
    auto wei_nhwc = create_tensor<CLTensor>(wei_shape_nhwc, dt, 1, QuantizationInfo(), data_layout);
    auto dst_nhwc = create_tensor<CLTensor>(dst_shape_nhwc, dt, 1, QuantizationInfo(), data_layout);

    TensorShape src_shape_nchw = src_shape_nhwc;
    TensorShape wei_shape_nchw = wei_shape_nhwc;
    TensorShape dst_shape_nchw = dst_shape_nhwc;

    permute(src_shape_nchw, PermutationVector(1U, 2U, 0U));
    permute(wei_shape_nchw, PermutationVector(1U, 2U, 0U, 3U));
    permute(dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    const PadStrideInfo conv_info = PadStrideInfo(1, 1, 0, 0);

    // Create indirect Convolution function
    CLIndirectConvolutionLayer conv{};
    conv.configure(&src_nhwc, &wei_nhwc, nullptr, &dst_nhwc, conv_info);

    src_nhwc.allocator()->allocate();
    wei_nhwc.allocator()->allocate();
    dst_nhwc.allocator()->allocate();

    library->fill_tensor_value(CLAccessor(src_nhwc), 1.f);
    library->fill_tensor_value(CLAccessor(wei_nhwc), 1.f);

    conv.run();

    // Compute reference to compare
    SimpleTensor<float> ref_src{ src_shape_nchw, dt };
    SimpleTensor<float> ref_wei{ wei_shape_nchw, dt };
    SimpleTensor<float> ref_bia{ bia_shape, dt };
    library->fill_tensor_value(ref_src, 1.f);
    library->fill_tensor_value(ref_wei, 1.f);
    // No bias
    library->fill_tensor_value(ref_bia, 0.f);
    auto ref_dst = reference::convolution_layer<float>(ref_src, ref_wei, ref_bia, dst_shape_nchw, conv_info);

    validate(CLAccessor(dst_nhwc), ref_dst);
}

/** Check whether the case of rectangle kernels i.e. when width and height of the weight_shape are not equal
 *  would lead to successful run
 */
TEST_CASE(NonSquareKernel, framework::DatasetMode::PRECOMMIT)
{
    const TensorShape    src_shape_nhwc = TensorShape(3U, 33U, 27U);
    const TensorShape    wei_shape_nhwc = TensorShape(3U, 5U, 7U, 4U); // non-square kernel
    const TensorShape    bia_shape      = TensorShape(4U);
    const TensorShape    dst_shape_nhwc = TensorShape(4U, 11U, 12U);
    constexpr DataType   dt             = DataType::F32;
    constexpr DataLayout data_layout    = DataLayout::NHWC;

    auto src_nhwc = create_tensor<CLTensor>(src_shape_nhwc, dt, 1, QuantizationInfo(), data_layout);
    auto wei_nhwc = create_tensor<CLTensor>(wei_shape_nhwc, dt, 1, QuantizationInfo(), data_layout);
    auto dst_nhwc = create_tensor<CLTensor>(dst_shape_nhwc, dt, 1, QuantizationInfo(), data_layout);

    TensorShape src_shape_nchw = src_shape_nhwc;
    TensorShape wei_shape_nchw = wei_shape_nhwc;
    TensorShape dst_shape_nchw = dst_shape_nhwc;

    permute(src_shape_nchw, PermutationVector(1U, 2U, 0U));
    permute(wei_shape_nchw, PermutationVector(1U, 2U, 0U, 3U));
    permute(dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    const PadStrideInfo conv_info = PadStrideInfo(3, 2, 1, 1, 2, 0, DimensionRoundingType::FLOOR);

    // Create indirect convolution function
    CLIndirectConvolutionLayer conv{};
    conv.configure(&src_nhwc, &wei_nhwc, nullptr, &dst_nhwc, conv_info);

    src_nhwc.allocator()->allocate();
    wei_nhwc.allocator()->allocate();
    dst_nhwc.allocator()->allocate();

    library->fill_tensor_value(CLAccessor(src_nhwc), 1.f);
    library->fill_tensor_value(CLAccessor(wei_nhwc), 1.f);

    conv.run();

    // Compute reference to compare
    SimpleTensor<float> ref_src{ src_shape_nchw, dt };
    SimpleTensor<float> ref_wei{ wei_shape_nchw, dt };
    SimpleTensor<float> ref_bia{ bia_shape, dt };
    library->fill_tensor_value(ref_src, 1.f);
    library->fill_tensor_value(ref_wei, 1.f);
    // No bias
    library->fill_tensor_value(ref_bia, 0.f);
    auto ref_dst = reference::convolution_layer<float>(ref_src, ref_wei, ref_bia, dst_shape_nchw, conv_info);

    validate(CLAccessor(dst_nhwc), ref_dst);
}
// *INDENT-OFF*
// clang-format off
// Note: Since the interface of indirect convolution is the same of direct convolution, we can reuse
// the direct convolution fixture
template <typename T>
using CLIndirectConvolutionLayerFixture = DirectConvolutionValidationFixture<CLTensor, CLAccessor, CLIndirectConvolutionLayer, T>;
template <typename T>
using CLIndirectConvolutionLayerMixedDataLayoutFixture = DirectConvolutionValidationFixture<CLTensor, CLAccessor, CLIndirectConvolutionLayer, T, true>;

TEST_SUITE(NHWC)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLIndirectConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT,
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

FIXTURE_DATA_TEST_CASE(RunLarge, CLIndirectConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
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
FIXTURE_DATA_TEST_CASE(RunSmall, CLIndirectConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
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
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLIndirectConvolutionLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT,
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
FIXTURE_DATA_TEST_CASE(RunLarge, CLIndirectConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
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
TEST_SUITE_END() // NHWC
TEST_SUITE_END() // IndirectConvolutionLayer
TEST_SUITE_END() // CL

} // namespace validation
} // namespace test
} // namespace arm_compute
