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
#include "arm_compute/runtime/CL/functions/CLCast.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "src/gpu/cl/kernels/ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/fixtures/GEMMLowpFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::opencl::kernels;

// Create function for CLGEMMReshapeRHSMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeOperator<opencl::kernels::ClGemmReshapeRhsMatrixKernel>;

// Create function for CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel
using CLGEMMLowpMatrixMultiplyReshapedOnlyRHS = CLSynthetizeOperator<opencl::kernels::ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel>;

// Fixture for CLGEMMLowpMatrixMultiplyReshapedOnlyRHS
using CLGEMMLowpMatrixMultiplyReshapedOnlyRHSMMULFixture =
    GEMMLowpMatrixMultiplyReshapedOnlyRHSMMULValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshapedOnlyRHS>;

// Fixture for CLGEMMLowpMatrixMultiplyReshapedOnlyRHS
using CLGEMMLowpMatrixMultiplyReshapedOnlyRHSMMULOutputStageFixtureSigned =
    GEMMLowpMatrixMultiplyReshapedOnlyRHSMMULOutputStageValidationFixture<int8_t, CLTensor, CLAccessor, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshapedOnlyRHS, CLReductionOperation, CLCast>;

using CLGEMMLowpMatrixMultiplyReshapedOnlyRHSMMULOutputStageFixtureUnsigned =
    GEMMLowpMatrixMultiplyReshapedOnlyRHSMMULOutputStageValidationFixture<uint8_t, CLTensor, CLAccessor, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshapedOnlyRHS, CLReductionOperation, CLCast>;

namespace
{
// *INDENT-OFF*
// clang-format off

/** M values to test */
const auto m_values = framework::dataset::make("M", {16, 49});

/** N values to test */
const auto n_values = framework::dataset::make("N", {16, 259});

/** K values to test */
const auto k_values = framework::dataset::make("K", {192});

/** Batch size values to test */
const auto b_values = framework::dataset::make("batch_size", {1, 2});

/** M0 values to test - Precommit */
const auto m0 = framework::dataset::make("M0", {1, 2, 4});

/** N0 values to test - Precommit */
const auto n0 = framework::dataset::make("N0", { 1, 4, 8});

/** K0 values to test - Precommit */
const auto k0 = framework::dataset::make("K0", { 4 });

/** H0 values to test - Precommit */
const auto h0 = framework::dataset::make("H0", 1);

/** Interleave values to test with RHS matrix */
const auto i_values_rhs = framework::dataset::make("interleave_rhs", { false });

/** Transpose values to test with RHS matrix */
const auto t_values_rhs = framework::dataset::make("transpose_rhs", { true });

const auto broadcast_bias = framework::dataset::make("broadcast_bias", {true, false});

} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMLowpMatrixMultiplyReshapedOnlyRhsMMUL)
FIXTURE_DATA_TEST_CASE(Signed, CLGEMMLowpMatrixMultiplyReshapedOnlyRHSMMULFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0),
                                                                   n0),
                                                                   k0),
                                                                   h0),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                    framework::dataset::make("DataType", { DataType::QASYMM8_SIGNED })))
{
    // Validate output
    if(arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()))
    {
        validate(CLAccessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
FIXTURE_DATA_TEST_CASE(Unsigned, CLGEMMLowpMatrixMultiplyReshapedOnlyRHSMMULFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0),
                                                                   n0),
                                                                   k0),
                                                                   h0),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                    framework::dataset::make("DataType", { DataType::QASYMM8})))
{
    // Validate output
    if(arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()))
    {
        validate(CLAccessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
FIXTURE_DATA_TEST_CASE(OutputStageSigned, CLGEMMLowpMatrixMultiplyReshapedOnlyRHSMMULOutputStageFixtureSigned, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0),
                                                                   n0),
                                                                   k0),
                                                                   h0),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   broadcast_bias),
                    framework::dataset::make("DataType", { DataType::QASYMM8_SIGNED})))
{
    // Validate output
    if(arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()))
    {
        validate(CLAccessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
FIXTURE_DATA_TEST_CASE(OutputStageUnsigned, CLGEMMLowpMatrixMultiplyReshapedOnlyRHSMMULOutputStageFixtureUnsigned, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0),
                                                                   n0),
                                                                   k0),
                                                                   h0),
                                                                   i_values_rhs),
                                                                   t_values_rhs),
                                                                   broadcast_bias),
                    framework::dataset::make("DataType", { DataType::QASYMM8})))
{
    // Validate output
    if(arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()))
    {
        validate(CLAccessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // GEMMLowpMatrixMultiplyReshapedOnlyRhsMMUL
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute