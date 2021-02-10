/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEDepthwiseConvolutionLayerNativeKernel.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DepthwiseConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

// Create function for NEDepthwiseConvolutionLayerKernel
using NEDepthwiseConvolutionLayerNative = NESynthetizeFunctionWithZeroConstantKernelBorder<NEDepthwiseConvolutionLayerNativeKernel>;

// Fixture for NEDepthwiseConvolutionLayerKernel
template <typename T>
using NEDepthwiseConvolutionLayerNativeFixture = DepthwiseConvolutionLayerNativeValidationFixture<Tensor, Accessor, NEDepthwiseConvolutionLayerNative, T>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);

/** Width values to test - Precommit */
const auto width_values_precommit = framework::dataset::make("width", { 17U } );

/** Width values to test - Nightly */
const auto width_values_nightly = framework::dataset::make("width", { 53U, 47U } );

/** Height values to test - Precommit */
const auto height_values_precommit = framework::dataset::make("height", { 19U } );

/** Height values to test - Nightly */
const auto height_values_nightly = framework::dataset::make("height", { 39U, 43U } );

/** Channel values to test - Precommit */
const auto channel_values_precommit = framework::dataset::make("channels", { 15U });

/** Channel values to test - Nightly */
const auto channel_values_nightly = framework::dataset::make("channels", { 33U, 19U });

/** Batch values to test - Precommit */
const auto batch_values_precommit = framework::dataset::make("batch", { 1U, 2U });

/** Batch values to test - Nightly */
const auto batch_values_nightly = framework::dataset::make("batch", { 1U, 3U });

/** Kernel size values to test - Precommit */
const auto kernel_sz_values_precommit = framework::dataset::make("kernel_size", { Size2D(1U, 1U), Size2D(1U, 3U) });

/** Kernel size values to test - Nightly */
const auto kernel_sz_values_nightly = framework::dataset::make("kernel_size", { Size2D(3U, 5U), Size2D(5U, 1U), Size2D(1U, 7U), Size2D(9U, 7U) });

/** Depth multiplier values to test - All */
const auto depth_multiplier_values = framework::dataset::make("depth_multiplier", { 1U, 3U });

/** Dilation values to test - All */
const auto dilation_values = framework::dataset::make("dilation", { Size2D(1U, 1U), Size2D(3U, 3U) });

/** Stride values to test - All */
const auto stride_values = framework::dataset::make("stride", { Size2D(1U, 1U), Size2D(3U, 2U) });

/** Padding values to test - All */
const auto padding_valid_values = framework::dataset::make("padding_valid", { true, false });

/** Data type values to test - All */
const auto data_type_values = framework::dataset::make("data_type", { DataType::F32 });

/** Data layout values to test - All */
const auto data_layout_values = framework::dataset::make("data_layout", { DataLayout::NHWC });
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DepthwiseConvolutionLayerNative)

TEST_CASE(ValidateNoPadding, framework::DatasetMode::ALL)
{
    // this test case will ensure that the kernel is not adding implicit padding
    constexpr uint32_t vector_size = 8; // Asummed vector size of the current native kernel
    constexpr auto     depth = vector_size * 2 + 1; // mis-aligned depth to force padding if exists.
    constexpr auto     data_layout = DataLayout::NHWC;
    constexpr auto     data_type = DataType::F32;

    const auto input_size  = Size2D{ 100, 100 }; // random plane size of the input
    const auto kernel_size = Size2D{ 4, 4 }; // random plane size of the kernel
    const auto pad_stride_info = PadStrideInfo(3, 3); // random convolution information to

    TensorShape src_shape{ depth, input_size.x(), input_size.y() };
    TensorShape weights_shape{ depth, kernel_size.x(), kernel_size.y() };
    TensorShape bias_shape{ depth };

    auto src     = create_tensor<Tensor>(src_shape, data_type, 1, QuantizationInfo(), data_layout);
    auto weights = create_tensor<Tensor>(weights_shape, data_type, 1, QuantizationInfo(), data_layout);
    auto biases  = create_tensor<Tensor>(bias_shape, data_type, 1, QuantizationInfo(), data_layout);
    auto dst     = create_tensor<Tensor>(TensorShape(), data_type, 1, QuantizationInfo(), data_layout);

    NEDepthwiseConvolutionLayerNativeKernel dwc;
    dwc.configure(&src, &weights, &biases, &dst, pad_stride_info);

    ARM_COMPUTE_EXPECT(src.info()->padding().empty(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->padding().empty(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(biases.info()->padding().empty(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->padding().empty(), framework::LogLevel::ERRORS);
}

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(width_values_precommit,
                                                                                                height_values_precommit),
                                                                                                channel_values_precommit),
                                                                                                batch_values_precommit),
                                                                                                kernel_sz_values_precommit),
                                                                                                depth_multiplier_values),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                data_type_values),
                                                                                                data_layout_values))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(width_values_nightly,
                                                                                                height_values_nightly),
                                                                                                channel_values_nightly),
                                                                                                batch_values_nightly),
                                                                                                kernel_sz_values_nightly),
                                                                                                depth_multiplier_values),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                data_type_values),
                                                                                                data_layout_values))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // DepthwiseConvolutionLayerNative
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
