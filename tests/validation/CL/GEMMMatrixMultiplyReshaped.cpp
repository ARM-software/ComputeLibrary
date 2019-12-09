/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyReshapedKernel.h"
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

// Create function for CLGEMMMatrixMultiplyReshapedKernel
using CLGEMMMatrixMultiplyReshaped = CLSynthetizeFunction<CLGEMMMatrixMultiplyReshapedKernel>;

// Fixture for CLGEMMMatrixMultiplyReshaped
template <typename T>
using CLGEMMMatrixMultiplyReshapedFixture = GEMMMatrixMultiplyReshapedValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped>;

// Fixture for CLGEMMMatrixMultiplyReshaped mixed precision
template <typename T>
using CLGEMMMatrixMultiplyReshapedMixedPrecisionFixture =
    GEMMMatrixMultiplyReshapedValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped, true>;

// Fixture for CLGEMMMatrixMultiplyReshaped3D
template <typename T>
using CLGEMMMatrixMultiplyReshaped3DFixture = GEMMMatrixMultiplyReshaped3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped>;

// Fixture for CLGEMMMatrixMultiplyReshaped3D mixed precision
template <typename T>
using CLGEMMMatrixMultiplyReshaped3DMixedPrecisionFixture =
    GEMMMatrixMultiplyReshaped3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped, true>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);

RelativeTolerance<float> rel_tolerance_f16_mixed_precision(0.001f);
constexpr float          abs_tolerance_f16_mixed_precision(0.01f);

RelativeTolerance<float> rel_tolerance_f16(0.001f);
constexpr float          abs_tolerance_f16(0.01f);

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

/** Alpha values to test - Precommit */
const auto a_values_precommit = framework::dataset::make("alpha", {-0.75f} );

/** Beta values to test - Precommit */
const auto beta_values_precommit = framework::dataset::make("beta", {-0.35f} );

/** M0 values to test - Precommit */
const auto m0_values_precommit = framework::dataset::make("M0", { 4 });

/** N0 values to test - Precommit */
const auto n0_values_precommit = framework::dataset::make("N0", { 4 });

/** K0 values to test - Precommit */
const auto k0_values_precommit = framework::dataset::make("K0", { 4 });

/** V0 values to test - Precommit */
const auto v0_values_precommit = framework::dataset::make("V0", 1, 3);

/** H0 values to test - Precommit */
const auto h0_values_precommit = framework::dataset::make("H0", 1, 3);

/** Alpha values to test - Nightly */
const auto a_values_nightly = framework::dataset::make("alpha", {1.0f} );

/** Beta values to test - Nightly */
const auto beta_values_nightly = framework::dataset::make("beta", {1.0f} );

/** M0 values to test - Nightly */
const auto m0_values_nightly = framework::dataset::make("M0", { 2, 3, 4, 8 });

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 2, 3, 4, 8 });

/** K0 values to test - Nightly */
const auto k0_values_nightly = framework::dataset::make("K0", { 2, 3, 4, 8 });

/** V0 values to test - Nightly */
const auto v0_values_nightly = framework::dataset::make("V0", 1, 4);

/** H0 values to test - Nightly */
const auto h0_values_nightly = framework::dataset::make("H0", 1, 4);

/** Interleave values to test with LHS matrix */
const auto i_values_lhs = framework::dataset::make("interleave_lhs", { true, false });

/** Interleave values to test with RHS matrix */
const auto i_values_rhs = framework::dataset::make("interleave_rhs", { true, false });

/** Broadcast bias from vector to matrix */
const auto broadcast_bias_values = framework::dataset::make("broadcast_bias", { false, true } );

/** LHS transposed values */
const auto lhs_transpose_values = framework::dataset::make("lhs_transpose", { false, true } );
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMMatrixMultiplyReshaped)
TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16

TEST_SUITE(MixedPrecision)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedMixedPrecisionFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedMixedPrecisionFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DMixedPrecisionFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DMixedPrecisionFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
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
                                                                   i_values_rhs),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}
TEST_SUITE_END() // MixedPrecision
TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMMMatrixMultiplyReshaped
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
