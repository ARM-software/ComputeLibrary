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
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeLHSMatrixKernel.h"
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

// Create function for CLGEMMReshapeLHSMatrixKernel
using CLGEMMReshapeLHSMatrix = CLSynthetizeFunction<CLGEMMReshapeLHSMatrixKernel>;

// Create function for CLGEMMReshapeRHSMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeFunction<CLGEMMReshapeRHSMatrixKernel>;

// Create function for CLGEMMMatrixMultiplyKernel
using CLGEMMMatrixMultiplyReshaped = CLSynthetizeFunction<CLGEMMMatrixMultiplyKernel>;

// Fixture for GEMMMatrixMultiplyInterleavedTransposedValidationFixture
template <typename T>
using CLGEMMMatrixMultiplyReshapedFixture =
    GEMMMatrixMultiplyInterleavedTransposedValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped>;

// Fixture for GEMMMatrixMultiplyInterleavedTransposed3DValidationFixture
template <typename T>
using CLGEMMMatrixMultiplyReshaped3DFixture =
    GEMMMatrixMultiplyInterleavedTransposed3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);

RelativeTolerance<half> rel_tolerance_f16(half(0.2));
constexpr float         tolerance_num_f16 = 0.02f;

/** Alpha values to test - Precommit */
const auto alpha_values = framework::dataset::make("alpha", {1.0f, -0.75f} );

/** Beta values to test - Precommit */
const auto beta_values = framework::dataset::make("beta", {-0.35f, 0.0f} );

/** M values to test - Precommit */
const auto m_values_precommit = framework::dataset::make("M", 37);

/** N values to test - Precommit */
const auto n_values_precommit = framework::dataset::make("N", 51);

/** K values to test - Precommit */
const auto k_values_precommit = framework::dataset::make("K", 23);

/** M values to test - Nightly */
const auto m_values_nightly = framework::dataset::make("M", {421, 1});

/** N values to test - Nightly */
const auto n_values_nightly = framework::dataset::make("N", 323);

/** K values to test - Nightly */
const auto k_values_nightly = framework::dataset::make("K", 207);

/** M_W values to test - Precommit */
const auto m_w_values_precommit = framework::dataset::make("M_W", 5);

/** M_H values to test - Precommit */
const auto m_h_values_precommit = framework::dataset::make("M_H", 7);

/** M_W values to test - Nightly */
const auto m_w_values_nightly = framework::dataset::make("M_W", 13);

/** M_H values to test - Nightly */
const auto m_h_values_nightly = framework::dataset::make("M_H", 27);

/** Batch size values to test */
const auto b_values = framework::dataset::make("batch_size", 1, 3);

/** Activation values to test */
const auto act_values = framework::dataset::make("Activation",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
});

/** V0 values to test - Precommit */
const auto v0_values_precommit = framework::dataset::make("V0", 2);

/** H0 values to test - Precommit */
const auto h0_values_precommit = framework::dataset::make("H0", 4);

/** V0 values to test - Nightly */
const auto v0_values_nightly = framework::dataset::make("V0", {2, 4});

/** H0 values to test - Nightly */
const auto h0_values_nightly = framework::dataset::make("H0", { 2, 4 });

/** Broadcast bias from vector to matrix */
const auto broadcast_bias_values = framework::dataset::make("broadcast_bias", {false, true} );

/** GPU architectures values to test */
const auto gpu_arch_values = framework::dataset::make("GPUArch",
{
    GPUTarget::MIDGARD,
    GPUTarget::BIFROST
});

/** Data types values to test in the configuration */
const auto data_type_values = framework::dataset::make("DataType",
{
    DataType::F32,
    DataType::F16
});

/** M values to test */
const auto fp16_mixed_precision_values = framework::dataset::make("fp16_mixed_precision", {true, false});

/** Configuration test */
void validate_configuration(unsigned int m_value, unsigned int n_value, unsigned int k_value, unsigned int b_value, unsigned int v0_value, unsigned int h0_value, bool broadcast_bias, bool fp16_mixed_precision, const ActivationLayerInfo &act_info, DataType data_type, GPUTarget gpu_arch_value)
{
    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = 4;
    lhs_info.k0         = 4;
    lhs_info.v0         = v0_value;
    lhs_info.interleave = true;
    lhs_info.transpose  = true;

    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = data_type == DataType::F32? 4 : 8;
    rhs_info.k0         = 1;
    rhs_info.h0         = h0_value;
    rhs_info.interleave = false;
    rhs_info.transpose  = false;

    GEMMReshapeInfo reshape_info(m_value, n_value, k_value, rhs_info.h0, lhs_info.v0, 0, false, broadcast_bias);

    const TensorShape lhs_shape(k_value, m_value, b_value);
    const TensorShape lhs_shape_reshaped = compute_lhs_reshaped_shape(TensorInfo(lhs_shape, 1, data_type),
                                                                      lhs_info,
                                                                      false);

    const TensorShape rhs_shape(n_value, k_value, b_value);
    const TensorShape rhs_shape_reshaped = compute_rhs_reshaped_shape(TensorInfo(rhs_shape, 1, data_type),
                                                                      rhs_info);

    const TensorShape dst_shape = compute_mm_shape(TensorInfo(lhs_shape_reshaped, 1, data_type),
                                                   TensorInfo(rhs_shape_reshaped, 1, data_type),
                                                   reshape_info);

    const TensorShape bias_shape(n_value,
                                 broadcast_bias? 1 : m_value,
                                 broadcast_bias? 1 : b_value);

    // Create tensors
    CLTensor lhs_reshaped = create_tensor<CLTensor>(lhs_shape_reshaped, data_type);
    CLTensor rhs_reshaped = create_tensor<CLTensor>(rhs_shape_reshaped, data_type);
    CLTensor bias         = create_tensor<CLTensor>(bias_shape, data_type);
    CLTensor dst          = create_tensor<CLTensor>(dst_shape, data_type);

    ARM_COMPUTE_EXPECT(lhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMMatrixMultiplyReshaped gemm;
    gemm.configure(gpu_arch_value, &lhs_reshaped, &rhs_reshaped, &bias, &dst, 1.0f, 2.0f, true, reshape_info, fp16_mixed_precision, act_info);
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMMatrixMultiplyInterleavedTransposed)
TEST_SUITE(Float)
TEST_SUITE(FP32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values_precommit,
                                                                   n_values_precommit),
                                                                   k_values_precommit),
                                                                   framework::dataset::make("batch_size", 1)),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   broadcast_bias_values),
                                                                   framework::dataset::make("fp16_mixed_precision", false)),
                                                                   act_values),
                                                                   data_type_values),
                                                                   gpu_arch_values),
m_value, n_value, k_value, b_value, v0_value, h0_value, broadcast_bias, fp16_mixed_precision_value, act_value, data_type_value, gpu_arch_value)
{
    validate_configuration(m_value, n_value, k_value, b_value, v0_value, h0_value, broadcast_bias, fp16_mixed_precision_value, act_value, data_type_value, gpu_arch_value);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values_precommit,
                                                                   n_values_precommit),
                                                                   k_values_precommit),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   broadcast_bias_values),
                                                                   framework::dataset::make("fp16_mixed_precision", false)),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values_nightly,
                                                                   n_values_nightly),
                                                                   k_values_nightly),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   broadcast_bias_values),
                                                                   framework::dataset::make("fp16_mixed_precision", false)),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values_precommit,
                                                                   m_h_values_precommit),
                                                                   n_values_precommit),
                                                                   k_values_precommit),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   broadcast_bias_values),
                                                                   framework::dataset::make("fp16_mixed_precision", false)),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values_nightly,
                                                                   m_h_values_nightly),
                                                                   n_values_nightly),
                                                                   k_values_nightly),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   broadcast_bias_values),
                                                                   framework::dataset::make("fp16_mixed_precision", false)),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values_precommit,
                                                                   n_values_precommit),
                                                                   k_values_precommit),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   broadcast_bias_values),
                                                                   fp16_mixed_precision_values),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values_nightly,
                                                                   n_values_nightly),
                                                                   k_values_nightly),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   broadcast_bias_values),
                                                                   fp16_mixed_precision_values),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values_precommit,
                                                                   m_h_values_precommit),
                                                                   n_values_precommit),
                                                                   k_values_precommit),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   broadcast_bias_values),
                                                                   fp16_mixed_precision_values),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values_nightly,
                                                                   m_h_values_nightly),
                                                                   n_values_nightly),
                                                                   k_values_nightly),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   broadcast_bias_values),
                                                                   fp16_mixed_precision_values),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16);
}

TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMMMatrixMulipltyInterleavedTransposed
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute