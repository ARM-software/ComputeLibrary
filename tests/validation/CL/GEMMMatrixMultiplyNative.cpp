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
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyNativeKernel.h"
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

// Create function for ClGemmMatrixMultiplyNativeKernel
using CLGEMMMatrixMultiplyNative = CLSynthetizeOperator<ClGemmMatrixMultiplyNativeKernel>;

// Fixture for CLGEMMMatrixMultiplyNative
template <typename T>
using CLGEMMMatrixMultiplyNativeFixture = GEMMMatrixMultiplyNativeValidationFixture<CLTensor, CLAccessor, T, CLGEMMMatrixMultiplyNative>;

// Fixture for CLGEMMMatrixMultiplyNative with post ops
template <typename T>
using CLGEMMMatrixMultiplyNativeWithPostOpsFixture =
    GEMMMatrixMultiplyNativeWithPostOpsValidationFixture<CLTensor, CLAccessor, T, CLGEMMMatrixMultiplyNative>;

// Fixture for CLGEMMMatrixMultiplyNative3D
template <typename T>
using CLGEMMMatrixMultiplyNative3DFixture = GEMMMatrixMultiplyNative3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMMatrixMultiplyNative>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);

/** Alpha values to test - Precommit */
const auto a_values = framework::dataset::make("alpha", {1.0f, -0.75f} );

/** Beta values to test - Precommit */
const auto beta_values = framework::dataset::make("beta", {-0.75f, 0.0f} );

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

/** M0 values to test - Precommit */
const auto m0_values_precommit = framework::dataset::make("M0", { 4, 6 });

/** N0 values to test - Precommit */
const auto n0_values_precommit = framework::dataset::make("N0", { 4 });

/** K0 values to test - Precommit */
const auto k0_values_precommit = framework::dataset::make("K0", { 4 });

/** H0 values to test - Precommit */
const auto h0_values_precommit = framework::dataset::make("H0", 1, 3);

/** M0 values to test - Nightly */
const auto m0_values_nightly = framework::dataset::make("M0", 1, 8);

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 2, 3, 4, 8 });

/** K0 values to test - Nightly */
const auto k0_values_nightly = framework::dataset::make("K0", { 2, 3, 4, 8 });

/** Broadcast bias from vector to matrix */
const auto broadcast_bias_values = framework::dataset::make("broadcast_bias", { false, true } );

/** Boundary handling cases for testing partial/non-partial (full) block dimensions, resulting from different combinations
 * of M, M0, N and N0 values.
 * M0 and N0 are kept constant, while the different test cases need to vary M and N.
 *
 * Eg. M = 64 and N = 33 result in a block dimension that has no partial blocks (all full blocks) in Y dimension and
 * parital blocks in X dimension.
 */
const auto boundary_handling_cases = combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                    // Large k to force potential out-of-bound reads on input0
                                    framework::dataset::make("K", 315),
                                    // Batch size == 1 to force potential out-of-bound reads on input0
                                    framework::dataset::make("batch_size", 1)),
                                    framework::dataset::make("M0", 4)),
                                    framework::dataset::make("N0", 4)),
                                    framework::dataset::make("K0", 4)),
                                    // Only need to test F32 as F16 shares identical boundary handling logics
                                    framework::dataset::make("DataType", DataType::F32)),
                                    framework::dataset::make("alpha", -0.75f )),
                                    framework::dataset::make("beta", -0.35f )),
                                    broadcast_bias_values),
                                    framework::dataset::make("Activation", ActivationLayerInfo()));

/** Post Ops */
using PostOpArgBroadcast =  CLGEMMMatrixMultiplyNativeWithPostOpsFixture<float>::PostOpArgBroadcast;
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
    // post_ops.push_back_op<experimental::PostOpAct<PostOpArgBroadcast>>(ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::RELU, 2.1F, 1.3F});
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<PostOpArgBroadcast>>(
        std::make_tuple(false, false, false),  // If broadcast in dims 0, 1 and 2
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
    return bool(ClGemmMatrixMultiplyNativeKernel::validate(&input0_info.clone()->set_is_resizable(true),
                                                          &input1_info.clone()->set_is_resizable(true),
                                                          &input2_info.clone()->set_is_resizable(true),
                                                          &output_info.clone()->set_is_resizable(true),1.f,1.f,
                                                          lhs_info,
                                                          rhs_info,
                                                          gemm_info));
}

/** Configuration test */
void validate_configuration(unsigned int m_value, unsigned int n_value, unsigned int k_value, unsigned int b_value, unsigned int m0_value, unsigned int n0_value, unsigned int k0_value, bool broadcast_bias, DataType data_type, const ActivationLayerInfo &act_info)
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

    GEMMKernelInfo kernel_info;
    kernel_info.m               = M;
    kernel_info.n               = N;
    kernel_info.k               = K;
    kernel_info.broadcast_bias  = broadcast_bias;
    kernel_info.activation_info = act_info;

    const TensorShape lhs_shape(K, M, b_value);
    const TensorShape rhs_shape(N, K, b_value);
    const TensorShape bias_shape(N,
                                 broadcast_bias? 1 : M,
                                 broadcast_bias? 1 : b_value);
    const TensorShape dst_shape = compute_mm_shape(TensorInfo(lhs_shape, 1, data_type),
                                                   TensorInfo(rhs_shape, 1, data_type),
                                                   kernel_info);

    // Create tensors
    CLTensor lhs  = create_tensor<CLTensor>(lhs_shape, data_type);
    CLTensor rhs  = create_tensor<CLTensor>(rhs_shape, data_type);
    CLTensor bias = create_tensor<CLTensor>(bias_shape, data_type);
    CLTensor dst  = create_tensor<CLTensor>(dst_shape, data_type);

    ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMMatrixMultiplyNative gemm;
    gemm.configure(lhs.info(), rhs.info(), bias.info(), dst.info(), 1.0f, 1.0f, lhs_info, rhs_info, kernel_info);
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMMatrixMultiplyNative)
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
    const unsigned int m = 1;
    const unsigned int n = 18;
    const unsigned int k = 13;
    const unsigned int batch = 2;
    TensorShape post_op_arg_shape(n, m + 1, batch); // output's Y dimension (m) is "widened", which is not allowed
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
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   framework::dataset::make("batch_size", 1)),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   broadcast_bias_values),
                                                                   act_values),
m_value, n_value, k_value, b_value, m0_value, n0_value, k0_value, broadcast_bias, act_value)
{
    validate_configuration(m_value, n_value, k_value, b_value, m0_value, n0_value, k0_value, broadcast_bias, DataType::F32, act_value);
}

FIXTURE_DATA_TEST_CASE(RunSmallBoundaryHandlingPartialInXPartialInY, CLGEMMMatrixMultiplyNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(
                        framework::dataset::make("M", 3),
                        framework::dataset::make("N", 1)),
                        boundary_handling_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallBoundaryHandlingPartialInXFullInY, CLGEMMMatrixMultiplyNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(
                        framework::dataset::make("M", 64),
                        framework::dataset::make("N", 51)),
                        boundary_handling_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallBoundaryHandlingFullInXFullInY, CLGEMMMatrixMultiplyNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(
                        framework::dataset::make("M", 64),
                        framework::dataset::make("N", 32)),
                        boundary_handling_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallBoundaryHandlingFullInXPartialInY, CLGEMMMatrixMultiplyNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(
                        framework::dataset::make("M", 37),
                        framework::dataset::make("N", 32)),
                        boundary_handling_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyNativeFixture<float>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyNative3DFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyNative3DFixture<float>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values),
                                                                   beta_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

TEST_SUITE(FusedPostOps)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyNativeWithPostOpsFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   framework::dataset::make("M0", { 4 })),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   framework::dataset::make("alpha", {1.0f} )),
                                                                   framework::dataset::make("beta", {1.0f} )),
                                                                   framework::dataset::make("broadcast_bias", { false, true } )),
                                                                   framework::dataset::make("Activation", { ActivationLayerInfo() })),
                                                                   post_op_lists)
                                                                   )
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

TEST_SUITE_END() //  FusedPostOps

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMMMatrixMulipltyNative
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
