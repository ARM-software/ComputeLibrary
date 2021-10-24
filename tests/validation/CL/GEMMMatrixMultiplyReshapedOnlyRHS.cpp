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
#include "src/core/experimental/PostOp.h"
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedOnlyRhsKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
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
using namespace arm_compute::opencl::kernels;

// Create function for ClGemmReshapeRhsMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeOperator<ClGemmReshapeRhsMatrixKernel>;

// Create function for ClGemmMatrixMultiplyReshapedOnlyRhsKernel
using CLGEMMMatrixMultiplyReshapedOnlyRHS = CLSynthetizeOperator<ClGemmMatrixMultiplyReshapedOnlyRhsKernel>;

// Fixture for CLGEMMMatrixMultiplyReshapedOnlyRHS
template <typename T>
using CLGEMMMatrixMultiplyReshapedOnlyRHSFixture = GEMMMatrixMultiplyReshapedOnlyRHSValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshapedOnlyRHS>;

// Fixture for CLGEMMMatrixMultiplyReshapedOnlyRHS3D
template <typename T>
using CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture = GEMMMatrixMultiplyReshapedOnlyRHS3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshapedOnlyRHS>;

// Fixture for CLGEMMMatrixMultiplyReshapedOnlyRHS with post ops
template <typename T>
using CLGEMMMatrixMultiplyReshapedOnlyRHSWithPostOpsFixture =
    GEMMMatrixMultiplyReshapedOnlyRHSWithPostOpsValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshapedOnlyRHS>;

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
const auto beta_values = framework::dataset::make("beta", {-0.35f} );

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
const auto b_values = framework::dataset::make("batch_size", 2);

/** Activation values to test */
const auto act_values = framework::dataset::make("Activation",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 10.f),
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

/** Boundary handling cases for testing partial/non-partial (full) block dimensions, resulting from different combinations
 * of M, M0, N and N0 values.
 * M0 and N0 are kept constant, while the different test cases need to vary M and N.
 *
 * Eg. M = 64 and N = 33 result in a block dimension that has no partial blocks (all full blocks) in Y dimension and
 * parital blocks in X dimension.
 */
const auto boundary_handling_cases = combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                    // Large k to force potential out-of-bound reads on input0
                                    framework::dataset::make("K", 315),
                                    // Batch size == 1 to force potential out-of-bound reads on input0
                                    framework::dataset::make("batch_size", 1)),
                                    framework::dataset::make("M0", 4)),
                                    framework::dataset::make("N0", 4)),
                                    framework::dataset::make("K0", 4)),
                                    framework::dataset::make("H0", 3)),
                                    i_values_rhs),
                                    t_values_rhs),
                                    framework::dataset::make("export_to_cl_image_rhs", {true, false})),
                                    // Only need to test F32 as F16 shares identical boundary handling logics
                                    framework::dataset::make("DataType", DataType::F32)),
                                    framework::dataset::make("alpha", -0.75f )),
                                    framework::dataset::make("beta", -0.35f )),
                                    broadcast_bias_values),
                                    framework::dataset::make("Activation", ActivationLayerInfo()));

/** Post Ops */
using PostOpArgBroadcast =  CLGEMMMatrixMultiplyReshapedOnlyRHSWithPostOpsFixture<float>::PostOpArgBroadcast;
experimental::PostOpList<PostOpArgBroadcast> post_ops_1()
{
    experimental::PostOpList<PostOpArgBroadcast> post_ops{};
    post_ops.push_back_op<experimental::PostOpAct<PostOpArgBroadcast>>(ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::LINEAR, 0.5F, 0.0F});
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<PostOpArgBroadcast>>(
        std::make_tuple(true, true, false),   // If broadcast in dims 0, 1 and 2
        0,
        ConvertPolicy::SATURATE);
    post_ops.push_back_op<experimental::PostOpAct<PostOpArgBroadcast>>(ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::RELU, 2.1F, 1.3F});
    return post_ops;
}
experimental::PostOpList<PostOpArgBroadcast> post_ops_2()
{
    experimental::PostOpList<PostOpArgBroadcast> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<PostOpArgBroadcast>>(
        std::make_tuple(false, true, true),   // If broadcast in dims 0, 1 and 2
        1,
        ConvertPolicy::SATURATE);
    post_ops.push_back_op<experimental::PostOpAct<PostOpArgBroadcast>>(ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::RELU, 2.1F, 1.3F});
    return post_ops;
}
experimental::PostOpList<PostOpArgBroadcast> post_ops_3()
{
    experimental::PostOpList<PostOpArgBroadcast> post_ops{};
    post_ops.push_back_op<experimental::PostOpAct<PostOpArgBroadcast>>(ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::RELU, 2.1F, 1.3F});
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<PostOpArgBroadcast>>(
        std::make_tuple(false, false, true),  // If broadcast in dims 0, 1 and 2
        1,
        ConvertPolicy::SATURATE);
    return post_ops;
}

/** Different Post Op Lists */
const auto post_op_lists = framework::dataset::make("post_op_lists", {
    post_ops_1(),
    post_ops_2(),
    post_ops_3(),
 } );

 bool is_post_op_list_valid(unsigned int m, unsigned int n, unsigned int k, unsigned int batch, DataType data_type, const experimental::PostOpList<ITensorInfo*>& post_ops)
{
    const auto lhs_info = GEMMLHSMatrixInfo(4,4,1,false,true);
    const auto rhs_info = GEMMRHSMatrixInfo(4,4,1,true,true,false);

    // Create TensorInfo for post op arguments
    TensorInfo input0_info(TensorShape(k, m, batch), 1, data_type);
    TensorInfo input1_info(TensorShape(n, k, batch), 1, data_type);
    TensorInfo input2_info(TensorShape(n), 1, data_type);
    TensorInfo output_info(TensorShape(n, m, batch), 1, data_type);

    const TensorInfo reshaped_input1_info = input1_info.clone()->set_tensor_shape(misc::shape_calculator::compute_rhs_reshaped_shape(input1_info, rhs_info));

    GEMMKernelInfo gemm_info(m, n, k, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
             false /**< reinterpret the input as 3D */,
             true  /**< Flag used to broadcast the bias addition */,
             false /**< wider accumm */,
             false /**< has pad y */,
           ActivationLayerInfo::ActivationFunction::IDENTITY,
             1   /**< Multiplication factor for the width of the 1xW transposed block */,
             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
             lhs_info,
             rhs_info,
             0  /**< Offset to be added to each element of the matrix A */,
             0 /**< Offset to be added to each element of the matrix B */,
             post_ops);
    return bool(ClGemmMatrixMultiplyReshapedOnlyRhsKernel::validate(&input0_info.clone()->set_is_resizable(true),
                                                          &reshaped_input1_info.clone()->set_is_resizable(true),
                                                          &input2_info.clone()->set_is_resizable(true),
                                                          &output_info.clone()->set_is_resizable(true),1.f,1.f,
                                                          lhs_info,
                                                          rhs_info,
                                                          gemm_info));
}
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
 *     - Correct F16 support for creating an OpenCL image object from buffer.
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
framework::dataset::make("Expected",            { false, false, false, false, false, false, false, true, false, true })),
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

TEST_SUITE(ValidateFusedPostOpsConfigs)
TEST_SUITE(Invalid)
TEST_CASE(UnsupportedPostOpSequence, framework::DatasetMode::ALL)
{
    const auto data_type = DataType::F32;
    const unsigned int m = 17;
    const unsigned int n = 1;
    const unsigned int k = 13;
    const unsigned int batch = 2;
    TensorShape post_op_arg0_shape(n, m, batch);
    TensorInfo post_op_arg_info(post_op_arg0_shape, 1, data_type);
    auto post_op_arg1_info = post_op_arg_info.clone();

    // Unsupported sequence of post ops
    experimental::PostOpList<ITensorInfo*> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo*>>(
        &post_op_arg_info,
        1,
        ConvertPolicy::SATURATE);
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo*>>(
        post_op_arg1_info.get(),
        0,
        ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid(m, n, k, batch, data_type, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_CASE(OutputWidened, framework::DatasetMode::ALL)
{
    // Invalid broadcast: post op tensors "widen" the output tensor
    const auto data_type = DataType::F32;
    const unsigned int m = 17;
    const unsigned int n = 1;
    const unsigned int k = 1;
    const unsigned int batch = 1;
    TensorShape post_op_arg_shape(n, m, batch + 4); // output's batch dimension is "widened", which is not allowed
    TensorInfo post_op_arg_info(post_op_arg_shape, 1, data_type);
    experimental::PostOpList<ITensorInfo*> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo*>>( &post_op_arg_info, 0, ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid(m, n, k, batch, data_type, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_CASE(BroadcastInXDimOnly, framework::DatasetMode::ALL)
{
    // Invalid broadcast: post op tensors broadcast in the first dimension (X) only
    const auto data_type = DataType::F32;
    const unsigned int m = 22;
    const unsigned int n = 16;
    const unsigned int k = 15;
    const unsigned int batch = 3;
    TensorShape post_op_arg_shape(1, m, batch);
    TensorInfo post_op_arg_info(post_op_arg_shape, 1, data_type);
    experimental::PostOpList<ITensorInfo*> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo*>>( &post_op_arg_info, 0, ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid(m, n, k, batch, data_type, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Invalid
TEST_SUITE(Valid)
TEST_CASE(EmptyPostOpList, framework::DatasetMode::ALL)
{
    const auto data_type = DataType::F32;
    const unsigned int m = 22;
    const unsigned int n = 16;
    const unsigned int k = 15;
    const unsigned int batch = 3;
    experimental::PostOpList<ITensorInfo*> post_ops{};

    ARM_COMPUTE_EXPECT(is_post_op_list_valid(m, n, k, batch, data_type, post_ops) == true, framework::LogLevel::ERRORS);
}
TEST_CASE(BroadcastInYDimOnly, framework::DatasetMode::ALL)
{
    const auto data_type = DataType::F32;
    const unsigned int m = 22;
    const unsigned int n = 16;
    const unsigned int k = 15;
    const unsigned int batch = 3;
    TensorShape post_op_arg_shape(n, 1, batch);
    TensorInfo post_op_arg_info(post_op_arg_shape, 1, data_type);
    experimental::PostOpList<ITensorInfo*> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo*>>( &post_op_arg_info, 0, ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid(m, n, k, batch, data_type, post_ops) == true, framework::LogLevel::ERRORS);
}
TEST_CASE(BroadcastInBothXandYDims, framework::DatasetMode::ALL)
{
    const auto data_type = DataType::F32;
    const unsigned int m = 22;
    const unsigned int n = 16;
    const unsigned int k = 15;
    const unsigned int batch = 3;
    TensorShape post_op_arg_shape(1, 1, batch);
    TensorInfo post_op_arg_info(post_op_arg_shape, 1, data_type);
    experimental::PostOpList<ITensorInfo*> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo*>>( &post_op_arg_info, 0, ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid(m, n, k, batch, data_type, post_ops) == true, framework::LogLevel::ERRORS);
}
TEST_CASE(BroadcastInAllDims, framework::DatasetMode::ALL)
{
    const auto data_type = DataType::F32;
    const unsigned int m = 22;
    const unsigned int n = 16;
    const unsigned int k = 15;
    const unsigned int batch = 3;
    TensorShape post_op_arg_shape(1, 1, 1);
    TensorInfo post_op_arg_info(post_op_arg_shape, 1, data_type);
    experimental::PostOpList<ITensorInfo*> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo*>>( &post_op_arg_info, 0, ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid(m, n, k, batch, data_type, post_ops) == true, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Valid
TEST_SUITE_END() // ValidateFusedPostOps
TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunPrecommitBoundaryHandlingPartialInXPartialInY, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(
                        framework::dataset::make("M", 3),
                        framework::dataset::make("N", 1)),
                        boundary_handling_cases))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunPrecommitBoundaryHandlingPartialInXFullInY, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(
                        framework::dataset::make("M", 64),
                        framework::dataset::make("N", 43)),
                        boundary_handling_cases))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunPrecommitBoundaryHandlingFullInXFullInY, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(
                        framework::dataset::make("M", 64),
                        framework::dataset::make("N", 32)),
                        boundary_handling_cases))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunPrecommitBoundaryHandlingFullInXPartialInY, CLGEMMMatrixMultiplyReshapedOnlyRHSFixture<float>, framework::DatasetMode::PRECOMMIT,
                combine(combine(
                        framework::dataset::make("M", 37),
                        framework::dataset::make("N", 32)),
                        boundary_handling_cases))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

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
                                                                   framework::dataset::make("export_to_cl_image_rhs", {false, true})),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
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
                                                                   framework::dataset::make("export_to_cl_image_rhs", {false, true})),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
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
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   framework::dataset::make("export_to_cl_image_rhs", {false, true})),
                                                                   framework::dataset::make("has_pad_y", {false, true})),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunNightly3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   framework::dataset::make("export_to_cl_image_rhs", {false, true})),
                                                                   framework::dataset::make("has_pad_y", {false, true})),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_SUITE(FusedPostOps)

FIXTURE_DATA_TEST_CASE(RunPrecommit, CLGEMMMatrixMultiplyReshapedOnlyRHSWithPostOpsFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   framework::dataset::make("H0", {1})),
                                                                   framework::dataset::make("interleave_rhs", { true })),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false, true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   framework::dataset::make("broadcast_bias", { false } )),
                                                                   act_values),
                                                                   post_op_lists)
                                                                   )
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_SUITE_END() //  FusedPostOps

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
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
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
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunPrecommit3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<half>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   framework::dataset::make("has_pad_y", {false, true})),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunNightly3D, CLGEMMMatrixMultiplyReshapedOnlyRHS3DFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   framework::dataset::make("has_pad_y", {false, true})),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE(FusedPostOps)

FIXTURE_DATA_TEST_CASE(RunPrecommit, CLGEMMMatrixMultiplyReshapedOnlyRHSWithPostOpsFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   framework::dataset::make("H0", {1})),
                                                                   framework::dataset::make("interleave_rhs", { true })),
                                                                   t_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values),
                                                                   beta_values),
                                                                   framework::dataset::make("broadcast_bias", { false } )),
                                                                   act_values),
                                                                   post_op_lists)
                                                                   )
{
    // Validate output only if the target platform supports the OpenCL cl_khr_image2d_from_buffer extension
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_SUITE_END() //  FusedPostOps

TEST_SUITE_END() // FP16

TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMMMatrixMulipltyReshapedOnlyRHS
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
