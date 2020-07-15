/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

// Create function for CLGEMMReshapeRHSMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeFunction<CLGEMMReshapeRHSMatrixKernel>;

// Create function for CLGEMMMatrixMultiplyReshapedOnlyRHSKernel
using CLGEMMMatrixMultiplyReshapedOnlyRHS = CLSynthetizeFunction<CLGEMMMatrixMultiplyReshapedOnlyRHSKernel>;

// Fixture for CLGEMMMatrixMultiplyReshapedOnlyRHS
template <typename T>
using CLGEMMMatrixMultiplyReshapedOnlyRHSFixture = GEMMMatrixMultiplyReshapedOnlyRHSValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshapedOnlyRHS>;

// Fixture for CLGEMMMatrixMultiplyReshapedOnlyRHS3D
template <typename T>
using CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture = GEMMMatrixMultiplyReshapedOnlyRHS3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshapedOnlyRHS>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);

RelativeTolerance<float> rel_tolerance_f16(0.001f);
constexpr float          abs_tolerance_f16(0.01f);

/** Alpha values to test */
const auto a_values = framework::dataset::make("alpha", {-0.75f} );

/** Beta values to test */
const auto beta_values = framework::dataset::make("beta", {-0.35f, 0.0f} );

/** M values to test */
const auto m_values = framework::dataset::make("M", 37);

/** M_W values to test */
const auto m_w_values = framework::dataset::make("M_W", 5);

/** M_H values to test */
const auto m_h_values = framework::dataset::make("M_H", 7);

/** N values to test */
const auto n_values = framework::dataset::make("N", 51);

/** K values to test */
const auto k_values = framework::dataset::make("K", 23);

/** Batch size values to test */
const auto b_values = framework::dataset::make("batch_size", 1, 3);

/** Activation values to test */
const auto act_values = framework::dataset::make("Activation",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
});

/** M0 values to test - precommit */
const auto m0_values_precommit = framework::dataset::make("M0", { 4 });

/** N0 values to test - precommit*/
const auto n0_values_precommit = framework::dataset::make("N0", { 4 });

/** K0 values to test - precommit*/
const auto k0_values_precommit = framework::dataset::make("K0", { 4 });

/** M0 values to test - nightly */
const auto m0_values_nightly = framework::dataset::make("M0", { 8 });

/** N0 values to test - nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 16 });

/** K0 values to test - nightly */
const auto k0_values_nightly = framework::dataset::make("K0", { 16 });

/** H0 values to test */
const auto h0_values = framework::dataset::make("H0", 1, 3);

/** Interleave values to test with RHS matrix */
const auto i_values_rhs = framework::dataset::make("interleave_rhs", { true, false });

/** Transpose values to test with RHS matrix */
const auto t_values_rhs = framework::dataset::make("transpose_rhs", { true, false });

/** Broadcast bias from vector to matrix */
const auto broadcast_bias_values = framework::dataset::make("broadcast_bias", { false, true } );

/** Configuration test */
bool validate_configuration(unsigned int m_value, unsigned int n_value, unsigned int k_value, unsigned int b_value,
                            unsigned int m0_value, unsigned int n0_value, unsigned int k0_value, unsigned int h0_value,
                            bool i_value_rhs, bool t_value_rhs, bool export_to_cl_image, bool broadcast_bias, bool input_as_3d, unsigned int depth_output_gemm3d, const ActivationLayerInfo &act_info,
                            DataType dt_input0, DataType dt_input1, DataType dt_input2, DataType dt_output, float alpha, float beta)
{
    const unsigned int M = m_value;
    const unsigned int N = n_value;
    const unsigned int K = k_value;

    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = m0_value;
    lhs_info.k0         = k0_value;

    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = n0_value;
    rhs_info.k0         = k0_value;
    rhs_info.h0         = h0_value;
    rhs_info.interleave = i_value_rhs;
    rhs_info.transpose  = t_value_rhs;
    rhs_info.export_to_cl_image = export_to_cl_image;

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = M;
    kernel_info.n                       = N;
    kernel_info.k                       = K;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = input_as_3d;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = act_info;

    const TensorShape lhs_shape(K, M, b_value);
    const TensorShape rhs_shape(N, K, b_value);
    const TensorShape rhs_shape_reshaped = compute_rhs_reshaped_shape(TensorInfo(rhs_shape, 1, dt_input1),
                                                                      rhs_info);

    const TensorShape dst_shape = compute_mm_shape(TensorInfo(lhs_shape, 1, dt_input0),
                                                   TensorInfo(rhs_shape_reshaped, 1, dt_input1),
                                                   kernel_info);

    const TensorShape bias_shape(N,
                                 M, // Correct calculation should be: broadcast_bias? 1 : M, it's wrong here on purpose just for validation test
                                 broadcast_bias? 1 : b_value);

    // Create tensor info
    TensorInfo lhs          = TensorInfo(lhs_shape, 1, dt_input0);
    TensorInfo rhs_reshaped = TensorInfo(rhs_shape_reshaped, 1, dt_input1);
    TensorInfo bias         = TensorInfo(bias_shape, 1, dt_input2);
    TensorInfo dst          = TensorInfo(dst_shape, 1, dt_output);

    // Create and configure function
    CLGEMMMatrixMultiplyReshapedOnlyRHS gemm;
    return bool(gemm.validate(&lhs, &rhs_reshaped, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info));
}

/** Zero padding test */
bool validate_zero_padding(unsigned int m_value, unsigned int n_value, unsigned int k_value, unsigned int b_value,
                            unsigned int m0_value, unsigned int n0_value, unsigned int k0_value, unsigned int h0_value,
                            bool i_value_rhs, bool t_value_rhs, bool export_to_cl_image, bool broadcast_bias, bool input_as_3d, unsigned int depth_output_gemm3d, const ActivationLayerInfo &act_info,
                            DataType dt_input0, DataType dt_input1, DataType dt_input2, DataType dt_output, float alpha, float beta)
{
    const unsigned int M = m_value;
    const unsigned int N = n_value;
    const unsigned int K = k_value;

    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = m0_value;
    lhs_info.k0         = k0_value;

    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = n0_value;
    rhs_info.k0         = k0_value;
    rhs_info.h0         = h0_value;
    rhs_info.interleave = i_value_rhs;
    rhs_info.transpose  = t_value_rhs;
    rhs_info.export_to_cl_image = export_to_cl_image;

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = M;
    kernel_info.n                       = N;
    kernel_info.k                       = K;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = input_as_3d;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = act_info;

    const TensorShape lhs_shape(K, M, b_value);
    const TensorShape rhs_shape(N, K, b_value);
    const TensorShape rhs_shape_reshaped = compute_rhs_reshaped_shape(TensorInfo(rhs_shape, 1, dt_input1),
                                                                      rhs_info);

    const TensorShape dst_shape = compute_mm_shape(TensorInfo(lhs_shape, 1, dt_input0),
                                                   TensorInfo(rhs_shape_reshaped, 1, dt_input1),
                                                   kernel_info);

    const TensorShape bias_shape(N,
                                 M, // Correct calculation should be: broadcast_bias? 1 : M, it's wrong here on purpose just for validation test
                                 broadcast_bias? 1 : b_value);

    // Create tensors
    CLTensor lhs  = create_tensor<CLTensor>(lhs_shape, dt_input0);
    CLTensor rhs_reshaped  = create_tensor<CLTensor>(rhs_shape_reshaped, dt_input1);
    CLTensor bias = create_tensor<CLTensor>(bias_shape, dt_input2);
    CLTensor dst  = create_tensor<CLTensor>(dst_shape, dt_output);

    ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Validate zero-padding
    CLGEMMMatrixMultiplyReshapedOnlyRHS gemm;

    gemm.configure(&lhs, &rhs_reshaped, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

    // Padding can be added along rhs and bias's X dimension
    return dst.info()->padding().empty() && lhs.info()->padding().empty() && bias.info()->padding().bottom == 0 && bias.info()->padding().top == 0;
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMMatrixMultiplyReshapedOnlyRHS)

/** Validate tests
 *
 * A series of validation tests on configurations which according to the API specification
 * the function should fail against.
 *
 * Checks performed in order:
 *     - Mismachting data type: input1, input2 and output need to have same data type as input0. Support data type: F32/F16.
 *     - Unsupported M0: MO can only be 1,2,3,4,5,6,7,8
 *     - Unsupported N0: NO can only be 2,3,4,8,16
 *     - Unsupported K0: KO can only be 2,3,4,8,16
 *     - Unsupported bias addition: bias broadcast mode is 0 if the input or output has to be reinterpreted as 3D
 *     - Incorrect bias diemension when bias broadcast mode is 1 and beta is not 0.0f, should be (n, 1), not (n, m)
 *     - Incorrect input0 dimension when input is reinterpreted as 3D: input0->dimension(1) * input0->dimension(2) != m
 *     - Correct support for creating an OpenCL image object from buffer
 *     - Incorrect support for creating an OpenCL image object from buffer. N0 is 2 but it can only be 4,8 and 16
 *     - Incorrect support for creating an OpenCL image object from buffer. Data type is F16 but it can only be F32
 */
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(
framework::dataset::make("batch_size",          { 1, 1, 1, 1, 1, 1, 2, 1, 1, 1 }),
framework::dataset::make("M0",                  { 4, 9, 4, 4, 4, 4, 4, 4, 4, 4 })),
framework::dataset::make("N0",                  { 4, 4, 18, 4, 4, 4, 4, 8, 2, 8 })),
framework::dataset::make("K0",                  { 4, 4, 4, 1, 4, 4, 4, 4, 4, 4 })),
framework::dataset::make("broadcast_bias",      { false, false, false, false, false, true, true, false, false, false })),
framework::dataset::make("input_as_3d",         { 0, 0, 0, 0, 1, 0, 1, 0, 0, 0 })),
framework::dataset::make("depth_output_gemm3d", { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 })),
framework::dataset::make("export_to_cl_image",  { false, false, false, false, false, false, false, true, true, true })),
framework::dataset::make("data_type_input0",    { DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F16})),
framework::dataset::make("data_type_input1",    { DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F16})),
framework::dataset::make("data_type_input2",    { DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F16})),
framework::dataset::make("data_type_output",    { DataType::F16, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F32, DataType::F16})),
framework::dataset::make("Beta",                { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f , 1.0f})),
framework::dataset::make("Expected",            { false, false, false, false, false, false, false, true, false, false })),
b_value, m0_value, n0_value, k0_value, broadcast_bias, input_as_3d, depth_output_gemm3d, export_to_cl_image, dt_input0, dt_intpu1, dt_input2, dt_output, beta, expected)
{
    bool expected_value = expected;

    // Change expected to false if the target platform does not support the OpenCL cl_khr_image2d_from_buffer extension
    if(!image2d_from_buffer_supported(CLKernelLibrary::get().get_device()) && export_to_cl_image)
    {
        expected_value = false;
    }

    bool status = validate_configuration(37, 51, 23, b_value, m0_value, n0_value, k0_value, 1, false, false, export_to_cl_image, broadcast_bias, input_as_3d, depth_output_gemm3d, ActivationLayerInfo(), dt_input0, dt_intpu1, dt_input2, dt_output, 1.0f, beta);
    ARM_COMPUTE_EXPECT(status == expected_value, framework::LogLevel::ERRORS);
}

/** Validate zero padding tests
 *
 * A series of validation tests to check that no padding is added as part of configuration for 4 different scenarios.
 *
 * Checks performed in order:
 *     - No partial blocks in both x and y dimensions
 *     - Partial blocks in x dimension
 *     - Partial blocks in y dimension
 *     - Partial blocks in both x and y dimensions
 */
DATA_TEST_CASE(ValidateZeroPadding, framework::DatasetMode::ALL, zip(zip(zip(zip(
framework::dataset::make("M",                   { 24, 64, 101, 1 }),
framework::dataset::make("N",                   { 48, 29, 16, 122 })),
framework::dataset::make("M0",                  { 4, 8, 7, 2 })),
framework::dataset::make("N0",                  { 4, 4, 16, 3 })),
framework::dataset::make("export_to_cl_image",  { false, true, true, false })),
m_value, n_value, m0_value, n0_value, export_to_cl_image)
{
    constexpr DataType dt = DataType::F32;
    // Disable export_to_cl_image if the target platform does not support the OpenCL cl_khr_image2d_from_buffer extension
    bool actual_export_to_cl_image = image2d_from_buffer_supported(CLKernelLibrary::get().get_device()) && export_to_cl_image;

    bool status = validate_zero_padding(m_value, n_value, 23, 1, m0_value, n0_value, 4, 1, false, false, actual_export_to_cl_image, false, 0, 0, ActivationLayerInfo(), dt, dt, dt, dt, 1.0f, 1.0f);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunPrecommit, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunNightly, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunPrecommit3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunNightly3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

TEST_SUITE(ExportToCLImage)
FIXTURE_DATA_TEST_CASE(RunPrecommit, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(image2d_from_buffer_supported(CLKernelLibrary::get().get_device()))
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunNightly, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(image2d_from_buffer_supported(CLKernelLibrary::get().get_device()))
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunPrecommit3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunNightly3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // ExportToCLImage
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunPrecommit, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<half>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunNightly, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunPrecommit3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<half>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunNightly3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

TEST_SUITE_END() // FP16

TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMMMatrixMulipltyReshapedOnlyRHS
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
