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
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyReshapedKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeLHSMatrixKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
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
using namespace arm_compute::misc::shape_calculator;

// Create function for CLGEMMReshapeLHSMatrixKernel
using CLGEMMReshapeLHSMatrix = CLSynthetizeFunction<CLGEMMReshapeLHSMatrixKernel>;

// Create function for CLGEMMReshapeRHSMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeFunction<CLGEMMReshapeRHSMatrixKernel>;

// Create function for CLGEMMMatrixMultiplyReshapedKernel
using CLGEMMLowpMatrixMultiplyReshaped = CLSynthetizeFunction<CLGEMMLowpMatrixMultiplyReshapedKernel>;

// Fixture for CLGEMMLowpMatrixMultiplyReshaped
using CLGEMMLowpMatrixMultiplyReshapedFixture = GEMMLowpMatrixMultiplyReshapedValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshaped>;

// Fixture for CLGEMMMatrixMultiplyReshaped3D
using CLGEMMLowpMatrixMultiplyReshaped3DFixture =
    GEMMLowpMatrixMultiplyReshaped3DValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMLowpMatrixMultiplyReshaped>;

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

/** V0 values to test - Precommit */
const auto v0_values_precommit = framework::dataset::make("V0", 1, 3);

/** H0 values to test - Precommit */
const auto h0_values_precommit = framework::dataset::make("H0", 1, 3);

/** M0 values to test - Nightly */
const auto m0_values_nightly = framework::dataset::make("M0", 2, 7);

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 2, 3, 4, 8 });

/** K0 values to test - Nightly */
const auto k0_values_nightly = framework::dataset::make("K0", { 2, 3, 4, 8, 16 });

/** V0 values to test - Nightly */
const auto v0_values_nightly = framework::dataset::make("V0", 1, 4);

/** H0 values to test - Nightly */
const auto h0_values_nightly = framework::dataset::make("H0", 1, 4);

/** Interleave values to test with LHS matrix */
const auto i_values_lhs = framework::dataset::make("interleave_lhs", { true, false });

/** Interleave values to test with RHS matrix */
const auto i_values_rhs = framework::dataset::make("interleave_rhs", { true, false });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMLowpMatrixMultiplyReshaped)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyReshapedFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyReshapedFixture, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMLowpMatrixMultiplyReshaped3DFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMLowpMatrixMultiplyReshaped3DFixture, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // GEMMLowpMatrixMultiplyReshaped
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute