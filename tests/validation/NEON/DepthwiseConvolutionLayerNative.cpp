/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolutionLayerNativeKernel.h"
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

/** Configuration test */
void validate_configuration(size_t width_value, size_t height_value, size_t channel_value, size_t batch_value, Size2D kernel_sz_value, size_t depth_multiplier_value, Size2D dilation_value, Size2D stride_value, bool padding_valid_value, DataType data_type_value, DataLayout data_layout_value)
{
    TensorShape src_shape(width_value, height_value, channel_value, batch_value);
    TensorShape weights_shape(kernel_sz_value.width, kernel_sz_value.height, channel_value * depth_multiplier_value);
    TensorShape biases_shape(channel_value * depth_multiplier_value);

    if(data_layout_value == DataLayout::NHWC)
    {
        permute(src_shape, PermutationVector(2U, 0U, 1U, 3U));
        permute(weights_shape, PermutationVector(2U, 0U, 1U));
    }

    TensorInfo src_info(src_shape, 1, data_type_value);
    TensorInfo weights_info(weights_shape, 1, data_type_value);
    TensorInfo biases_info(biases_shape, 1, data_type_value);

    src_info.set_data_layout(data_layout_value);
    weights_info.set_data_layout(data_layout_value);
    biases_info.set_data_layout(data_layout_value);

    PadStrideInfo conv_info;
    if(padding_valid_value)
    {
        conv_info = PadStrideInfo();
    }
    else
    {
        conv_info = calculate_same_pad(src_shape, weights_shape, PadStrideInfo(stride_value.width, stride_value.height), data_layout_value, dilation_value);
    }

    const TensorShape dst_shape = compute_depthwise_convolution_shape(src_info, weights_info, conv_info, depth_multiplier_value, dilation_value);

    // Create tensors
    Tensor src      = create_tensor<Tensor>(src_shape, data_type_value, 1, QuantizationInfo(), data_layout_value);
    Tensor weights  = create_tensor<Tensor>(weights_shape, data_type_value, 1, QuantizationInfo(), data_layout_value);
    Tensor biases   = create_tensor<Tensor>(biases_shape, data_type_value, 1, QuantizationInfo(), data_layout_value);
    Tensor dst      = create_tensor<Tensor>(dst_shape, data_type_value, 1, QuantizationInfo(), data_layout_value);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(biases.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEDepthwiseConvolutionLayerNative dwc;
    dwc.configure(&src, &weights, &biases, &dst, conv_info, depth_multiplier_value, dilation_value);
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DepthwiseConvolutionLayerNative)
TEST_SUITE(Float)
TEST_SUITE(FP32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(width_values_precommit,
                                                                                                                                           height_values_precommit),
                                                                                                                                           channel_values_precommit),
                                                                                                                                           batch_values_precommit),
                                                                                                                                           kernel_sz_values_precommit),
                                                                                                                                           depth_multiplier_values),
                                                                                                                                           dilation_values),
                                                                                                                                           stride_values),
                                                                                                                                           padding_valid_values),
                                                                                                                                           data_type_values),
                                                                                                                                           data_layout_values),
width_value, height_value, channel_value, batch_value, kernel_sz_value, depth_multiplier_value, dilation_value, stride_value, padding_valid_value, data_type_value, data_layout_value)
{
    validate_configuration(width_value, height_value, channel_value, batch_value, kernel_sz_value, depth_multiplier_value, dilation_value, stride_value, padding_valid_value, data_type_value, data_layout_value);
}

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
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute