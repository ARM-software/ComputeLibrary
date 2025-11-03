/*
 * Copyright (c) 2019-2021, 2025 Arm Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/gpu/cl/kernels/ClGemmLowpMatrixMultiplyReshapedKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeLhsMatrixKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
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
using framework::dataset::make;
using namespace arm_compute::misc::shape_calculator;

// Create function for ClGemmReshapeLhsMatrixKernel
using CLGEMMReshapeLHSMatrix = CLSynthetizeOperator<opencl::kernels::ClGemmReshapeLhsMatrixKernel>;

// Create function for ClGemmReshapeRhsMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeOperator<opencl::kernels::ClGemmReshapeRhsMatrixKernel>;

// Create function for CLGEMMLowpMatrixMultiplyReshapedKernel
using CLGEMMLowpMatrixMultiplyReshaped = CLSynthetizeOperator<opencl::kernels::ClGemmLowpMatrixMultiplyReshapedKernel>;

// Fixture for CLGEMMLowpMatrixMultiplyReshaped
using CLGEMMLowpMatrixMultiplyReshapedFixture = GEMMLowpMatrixMultiplyReshapedValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshaped>;

// Fixture for CLGEMMMatrixMultiplyReshaped3D
using CLGEMMLowpMatrixMultiplyReshaped3DFixture =
    GEMMLowpMatrixMultiplyReshaped3DValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshaped>;

namespace
{
// *INDENT-OFF*
// clang-format off

/** M, N combinations to test
 *  1: Special 1x1 case
 *  2: Special multples of processor size in both dimensions
 *  3: Non multiples of processor size in both dimensions
*/
const auto m_n_values = zip(
    make("M", {1, 16, 37}),
    make("N", {1, 16, 51})
    );

/** M values to test */
const auto m_values = make("M", {1, 37});

/** M_W values to test */
const auto m_w_values = make("M_W", 5);

/** M_H values to test */
const auto m_h_values = make("M_H", 7);

/** N values to test */
const auto n_values = make("N", {1, 51});

/** K values to test */
const auto k_values = make("K", 23);

/** Batch size values to test */
const auto b_values = make("batch_size", 1, 3);

/** M0 values to test - Precommit */
const auto m0_values_precommit_1 = make("M0", { 4 });
const auto m0_values_precommit_2 = make("M0", { 6 });

/** N0 values to test - Precommit */
const auto n0_values_precommit = make("N0", { 4 });

/** K0 values to test - Precommit */
const auto k0_values_precommit = make("K0", { 16 });

/** V0 values to test - Precommit */
const auto v0_values_precommit = make("V0", 1, 3);

/** H0 values to test - Precommit */
const auto h0_values_precommit = make("H0", 1, 3);

/** M0 values to test - Nightly */
const auto m0_values_nightly = make("M0", 2, 7);

/** N0 values to test - Nightly */
const auto n0_values_nightly = make("N0", { 2, 3, 4, 8 });

/** K0 values to test - Nightly */
const auto k0_values_nightly = make("K0", { 2, 3, 4, 8, 16 });

/** V0 values to test - Nightly */
const auto v0_values_nightly = make("V0", 1, 4);

/** H0 values to test - Nightly */
const auto h0_values_nightly = make("H0", 1, 4);

/** Interleave values to test with LHS matrix */
const auto i_values_lhs = make("interleave_lhs", { true, false });

/** Interleave values to test with RHS matrix */
const auto i_values_rhs = make("interleave_rhs", { true, false });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMLowpMatrixMultiplyReshaped)

TEST_SUITE(QUANTIZED)

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyReshapedFixture, framework::DatasetMode::ALL,
                combine(m_n_values,
                                                                   k_values,
                                                                   b_values,
                                                                   m0_values_precommit_1,
                                                                   n0_values_precommit,
                                                                   k0_values_precommit,
                                                                   v0_values_precommit,
                                                                   h0_values_precommit,
                                                                   i_values_lhs,
                                                                   i_values_rhs,
                                                                   make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyReshapedFixture, framework::DatasetMode::DISABLED,
                combine(m_values,
                                                                   n_values,
                                                                   k_values,
                                                                   b_values,
                                                                   m0_values_nightly,
                                                                   n0_values_nightly,
                                                                   k0_values_nightly,
                                                                   v0_values_nightly,
                                                                   h0_values_nightly,
                                                                   i_values_lhs,
                                                                   i_values_rhs,
                                                                   make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMLowpMatrixMultiplyReshaped3DFixture, framework::DatasetMode::ALL,
                combine(m_w_values,
                                                                   m_h_values,
                                                                   n_values,
                                                                   k_values,
                                                                   b_values,
                                                                   m0_values_precommit_1,
                                                                   n0_values_precommit,
                                                                   k0_values_precommit,
                                                                   v0_values_precommit,
                                                                   h0_values_precommit,
                                                                   i_values_lhs,
                                                                   i_values_rhs,
                                                                   make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMLowpMatrixMultiplyReshaped3DFixture, framework::DatasetMode::DISABLED,
                combine(m_w_values,
                                                                   m_h_values,
                                                                   n_values,
                                                                   k_values,
                                                                   b_values,
                                                                   m0_values_nightly,
                                                                   n0_values_nightly,
                                                                   k0_values_nightly,
                                                                   v0_values_nightly,
                                                                   h0_values_nightly,
                                                                   i_values_lhs,
                                                                   i_values_rhs,
                                                                   make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyReshapedFixture, framework::DatasetMode::ALL,
                combine(m_n_values,
                                                                   k_values,
                                                                   b_values,
                                                                   m0_values_precommit_2,
                                                                   n0_values_precommit,
                                                                   k0_values_precommit,
                                                                   v0_values_precommit,
                                                                   h0_values_precommit,
                                                                   i_values_lhs,
                                                                   i_values_rhs,
                                                                   make("DataType", { DataType::QASYMM8_SIGNED })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMLowpMatrixMultiplyReshaped3DFixture, framework::DatasetMode::ALL,
                combine(m_w_values,
                                                                   m_h_values,
                                                                   n_values,
                                                                   k_values,
                                                                   b_values,
                                                                   m0_values_precommit_2,
                                                                   n0_values_precommit,
                                                                   k0_values_precommit,
                                                                   v0_values_precommit,
                                                                   h0_values_precommit,
                                                                   i_values_lhs,
                                                                   i_values_rhs,
                                                                   make("DataType", { DataType::QASYMM8_SIGNED })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // QUANTIZED
TEST_SUITE_END() // GEMMLowpMatrixMultiplyReshaped
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
