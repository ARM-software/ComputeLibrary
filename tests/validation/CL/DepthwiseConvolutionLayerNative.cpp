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
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayerNativeKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/framework/Asserts.h"
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

// Create function for CLDepthwiseConvolutionLayerNativeKernel
using CLDepthwiseConvolutionLayerNative = CLSynthetizeFunction<CLDepthwiseConvolutionLayerNativeKernel>;

// Fixture for CLDepthwiseConvolutionLayerNative
template <typename T>
using CLDepthwiseConvolutionLayerNativeFixture = DepthwiseConvolutionLayerNativeConfigurableValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayerNative, T>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);

RelativeTolerance<half_float::half>  rel_tolerance_f16(half_float::half(0.01f));
constexpr float                      abs_tolerance_f16(0.03f);

/** Width values to test - Precommit */
const auto width_values_precommit = framework::dataset::make("width", { 1U, 33U } );

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

/** Channel values to test with cl_image support - Precommit */
const auto channel_values_export_to_cl_image_precommit = framework::dataset::make("channels", { 16U });

/** Channel values to test with cl_image support - Nightly */
const auto channel_values_export_to_cl_image_nightly = framework::dataset::make("channels", { 32U });

/** Batch values to test - Precommit */
const auto batch_values_precommit = framework::dataset::make("batch", { 1U, 2U });

/** Batch values to test - Nightly */
const auto batch_values_nightly = framework::dataset::make("batch", { 1U, 3U });

/** Kernel size values to test - Precommit */
const auto kernel_sz_values_precommit = framework::dataset::make("kernel_size", { Size2D(1U, 1U), Size2D(1U, 3U), Size2D(5U, 5U) });

/** Kernel size values to test - Nightly */
const auto kernel_sz_values_nightly = framework::dataset::make("kernel_size", { Size2D(3U, 5U), Size2D(5U, 1U), Size2D(1U, 7U), Size2D(9U, 7U) });

/** Depth multiplier values to test - All */
const auto depth_multiplier_values = framework::dataset::make("depth_multiplier", {3U});

/** Dilation values to test - All */
const auto dilation_values = framework::dataset::make("dilation", { Size2D(1U, 1U), Size2D(3U, 3U) });

/** Stride values to test - All */
const auto stride_values = framework::dataset::make("stride", { Size2D(1U, 1U), Size2D(3U, 2U) });

/** Padding values to test - All */
const auto padding_valid_values = framework::dataset::make("padding_valid", { true, false });

/** Data type values to test - All */
const auto data_type_values = framework::dataset::make("data_type", { DataType::F32, DataType::F16 });

/** Data layout values to test - All */
const auto data_layout_values = framework::dataset::make("data_layout", { DataLayout::NHWC });

/** N0 values to test - Precommit */
const auto n0_values_precommit = framework::dataset::make("N0", {2, 4});

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", {3, 8});

/** N0 values to test with cl_image support - Precommit */
const auto n0_values_export_to_cl_image_precommit = framework::dataset::make("N0", {4});

/** N0 values to test with cl_image support - Nightly */
const auto n0_values_export_to_cl_image_nightly = framework::dataset::make("N0", {8});

/** Activation values to test */
const auto act_values = framework::dataset::make("Activation",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.0f, 0.5f),
});

} // namespace

TEST_SUITE(CL)
TEST_SUITE(DepthwiseConvolutionLayerNative)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_precommit,
                                                                                                height_values_precommit),
                                                                                                channel_values_precommit),
                                                                                                batch_values_precommit),
                                                                                                kernel_sz_values_precommit),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_precommit),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_nightly,
                                                                                                height_values_nightly),
                                                                                                channel_values_nightly),
                                                                                                batch_values_nightly),
                                                                                                kernel_sz_values_nightly),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_nightly),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

TEST_SUITE(ExportWeightsToCLImage)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_precommit,
                                                                                                height_values_precommit),
                                                                                                channel_values_export_to_cl_image_precommit),
                                                                                                batch_values_precommit),
                                                                                                kernel_sz_values_precommit),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_export_to_cl_image_precommit),
                                                                                                framework::dataset::make("ExportToCLImage", true)))
{
   // Validate output
    if(_validate_output)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_nightly,
                                                                                                height_values_nightly),
                                                                                                channel_values_export_to_cl_image_nightly),
                                                                                                batch_values_nightly),
                                                                                                kernel_sz_values_nightly),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_export_to_cl_image_nightly),
                                                                                                framework::dataset::make("ExportToCLImage", true)))
{
   // Validate output
    if(_validate_output)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // ExportWeightsToCLImage
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerNativeFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_precommit,
                                                                                                height_values_precommit),
                                                                                                channel_values_precommit),
                                                                                                batch_values_precommit),
                                                                                                kernel_sz_values_precommit),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F16)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_precommit),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
        validate(CLAccessor(_target), _reference, rel_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerNativeFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_nightly,
                                                                                                height_values_nightly),
                                                                                                channel_values_nightly),
                                                                                                batch_values_nightly),
                                                                                                kernel_sz_values_nightly),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F16)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_nightly),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}
TEST_SUITE(ExportWeightsToCLImage)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerNativeFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_precommit,
                                                                                                height_values_precommit),
                                                                                                channel_values_export_to_cl_image_precommit),
                                                                                                batch_values_precommit),
                                                                                                kernel_sz_values_precommit),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F16)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_export_to_cl_image_precommit),
                                                                                                framework::dataset::make("ExportToCLImage", true)))
{
   // Validate output
    if(_validate_output)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerNativeFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_nightly,
                                                                                                height_values_nightly),
                                                                                                channel_values_export_to_cl_image_nightly),
                                                                                                batch_values_nightly),
                                                                                                kernel_sz_values_nightly),
                                                                                                framework::dataset::make("depth_multiplier", 1)),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F16)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                n0_values_export_to_cl_image_nightly),
                                                                                                framework::dataset::make("ExportToCLImage", true)))
{
   // Validate output
    if(_validate_output)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // ExportWeightsToCLImage
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE(DepthMultiplier)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_precommit,
                                                                                                height_values_precommit),
                                                                                                channel_values_precommit),
                                                                                                batch_values_precommit),
                                                                                                kernel_sz_values_precommit),
                                                                                                depth_multiplier_values),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                framework::dataset::make("N0", 1)),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerNativeFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_nightly,
                                                                                                height_values_nightly),
                                                                                                channel_values_nightly),
                                                                                                batch_values_nightly),
                                                                                                kernel_sz_values_nightly),
                                                                                                depth_multiplier_values),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                framework::dataset::make("N0", 1)),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall, CLDepthwiseConvolutionLayerNativeFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_precommit,
                                                                                                height_values_precommit),
                                                                                                channel_values_precommit),
                                                                                                batch_values_precommit),
                                                                                                kernel_sz_values_precommit),
                                                                                                depth_multiplier_values),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F16)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                framework::dataset::make("N0", 1)),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
        validate(CLAccessor(_target), _reference, rel_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE_NEW(RunLarge, CLDepthwiseConvolutionLayerNativeFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                width_values_nightly,
                                                                                                height_values_nightly),
                                                                                                channel_values_nightly),
                                                                                                batch_values_nightly),
                                                                                                kernel_sz_values_nightly),
                                                                                                depth_multiplier_values),
                                                                                                dilation_values),
                                                                                                stride_values),
                                                                                                padding_valid_values),
                                                                                                framework::dataset::make("DataType", DataType::F16)),
                                                                                                data_layout_values),
                                                                                                act_values),
                                                                                                framework::dataset::make("N0", 1)),
                                                                                                framework::dataset::make("ExportToCLImage", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // DepthMultiplier
TEST_SUITE_END() // DepthwiseConvolutionLayerNative
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
