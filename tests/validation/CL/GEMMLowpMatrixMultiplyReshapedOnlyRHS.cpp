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
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
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
#include "tests/validation/fixtures/GEMMLowpFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

// Create function for CLGEMMReshapeRHSMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeFunction<CLGEMMReshapeRHSMatrixKernel>;

// Create function for CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel
using CLGEMMLowpMatrixMultiplyReshapedOnlyRHS = CLSynthetizeFunction<CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel>;

// Fixture for CLGEMMLowpMatrixMultiplyReshapedOnlyRHS
using CLGEMMLowpMatrixMultiplyReshapedOnlyRHSFixture = GEMMLowpMatrixMultiplyReshapedOnlyRHSValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshapedOnlyRHS>;

// Fixture for CLGEMMLowpMatrixMultiplyReshapedOnlyRHS3D
using CLGEMMLowpMatrixMultiplyReshapedOnlyRHS3DFixture =
    GEMMLowpMatrixMultiplyReshapedOnlyRHS3DValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshapedOnlyRHS>;

namespace
{
// *INDENT-OFF*
// clang-format off

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

/** M0 values to test - Precommit */
const auto m0_values_precommit = framework::dataset::make("M0", {4, 6});

/** N0 values to test - Precommit */
const auto n0_values_precommit = framework::dataset::make("N0", { 4 });

/** K0 values to test - Precommit */
const auto k0_values_precommit = framework::dataset::make("K0", { 16 });

/** H0 values to test - Precommit */
const auto h0_values_precommit = framework::dataset::make("H0", 1, 3);

/** M0 values to test - Nightly */
const auto m0_values_nightly = framework::dataset::make("M0", 2, 8);

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 2, 3, 4, 8 });

/** K0 values to test - Nightly */
const auto k0_values_nightly = framework::dataset::make("K0", { 2, 3, 4, 8, 16 });

/** H0 values to test - Nightly */
const auto h0_values_nightly = framework::dataset::make("H0", 1, 4);

/** Interleave values to test with RHS matrix */
const auto i_values_rhs = framework::dataset::make("interleave_rhs", { true, false });

/** Transpose values to test with RHS matrix */
const auto t_values_rhs = framework::dataset::make("transpose_rhs", { true });

/** Configuration test */
void validate_configuration(unsigned int m_value, unsigned int n_value, unsigned int k_value, unsigned int b_value, unsigned int m0_value, unsigned int n0_value, unsigned int k0_value, unsigned int h0_value, bool i_value_rhs)
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
    rhs_info.transpose  = true;

    GEMMReshapeInfo gemm_info(M, N, K);

    const TensorShape lhs_shape(K, M, b_value);
    const TensorShape rhs_shape(N, K, b_value);
    const TensorShape rhs_shape_reshaped = compute_rhs_reshaped_shape(TensorInfo(rhs_shape, 1, DataType::QASYMM8),
                                                                      rhs_info);

    const TensorShape dst_shape = compute_mm_shape(TensorInfo(lhs_shape, 1, DataType::QASYMM8),
                                                   TensorInfo(rhs_shape_reshaped, 1, DataType::QASYMM8),
                                                   gemm_info);

    // Create tensors
    CLTensor lhs          = create_tensor<CLTensor>(lhs_shape, DataType::QASYMM8);
    CLTensor rhs_reshaped = create_tensor<CLTensor>(rhs_shape_reshaped, DataType::QASYMM8);
    CLTensor dst          = create_tensor<CLTensor>(dst_shape, DataType::S32);

    ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMLowpMatrixMultiplyReshapedOnlyRHS gemm;
    gemm.configure(&lhs, &rhs_reshaped, &dst, lhs_info, rhs_info, gemm_info);
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMLowpMatrixMultiplyReshapedOnlyRHS)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   framework::dataset::make("batch_size", 1)),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_rhs),
m_value, n_value, k_value, b_value, m0_value, n0_value, k0_value, h0_value, i_value_rhs)
{
    validate_configuration(m_value, n_value, k_value, b_value, m0_value, n0_value, k0_value, h0_value, i_value_rhs);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyReshapedOnlyRHSFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_rhs),
                                                                   t_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyReshapedOnlyRHSFixture, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_rhs),
                                                                   t_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMLowpMatrixMultiplyReshapedOnlyRHS3DFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_rhs),
                                                                   t_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMLowpMatrixMultiplyReshapedOnlyRHS3DFixture, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_rhs),
                                                                   t_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // GEMMLowpMatrixMultiplyReshapedOnlyRHS
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute