/*
 * Copyright (c) 2022, 2025-2026 Arm Limited.
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

#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/fixtures/GEMMFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;
using namespace arm_compute::opencl::kernels;

// Create function for ClGemmReshapeRhsMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeOperator<ClGemmReshapeRhsMatrixKernel>;

// Create function for ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel
using CLGEMMMatrixMultiplyReshapedOnlyRhsMMUL = CLSynthetizeOperator<ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel>;

// Fixture for CLGEMMMatrixMultiplyReshapedOnlyRhsMMUL
template <typename T>
using CLGEMMMatrixMultiplyReshapedOnlyRhsMMULFixture =
    GEMMMatrixMultiplyReshapedOnlyRhsMMULValidationFixture<CLTensor,
                                                           CLAccessor,
                                                           T,
                                                           CLGEMMReshapeRHSMatrix,
                                                           CLGEMMMatrixMultiplyReshapedOnlyRhsMMUL>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);
RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.001f));
constexpr float          abs_tolerance_f16(0.3f);

/** Alpha values to test - Precommit */
const auto a_values = make("alpha", {1.0f, 0.75f} );

/** Beta values to test - Precommit */
const auto beta_values = make("beta", {0.0f, -0.75f} );

/** M values to test */
const auto m_values = make("M", {49});

/** N values to test */
const auto n_values              = make("N", {257, 64, 48});
const auto n_values_fp16         = make("N", {65, 80});
const auto n_values_texture_fp16 = make("N", {128, 96, 48});

/** K values to test */
/** The test case requires this to be multiple of 4*/
const auto k_values = make("K", {192});
const auto k_values_fp16 = make("K", {64});

/** Batch size values to test */
const auto b_values = make("batch_size", {1, 2});

/** Activation values to test */
const auto act_values = make("Activation",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ELU)
});

/** M0 values to test - Precommit */
const auto m0_values_precommit = make("M0", { 1, 2, 4 });
const auto m0_values_precommit_fp16 = make("M0", { 1, 2, 3, 4, 8 });

/** N0 values to test - Precommit */
const auto n0_values_precommit              = make("N0", { 4, 8 });
const auto n0_values_precommit_fp16         = make("N0", { 2, 4, 8, 16 });
const auto n0_values_precommit_texture_fp16 = make("N0", { 4, 8 });

/** K0 values to test - Precommit */
const auto k0_values_precommit = make("K0", { 1 });

/** Broadcast bias from vector to matrix */
const auto broadcast_bias_values = make("broadcast_bias", { false, true } );

} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMMatrixMultiplyReshapedOnlyRhsMMUL)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedOnlyRhsMMULFixture<float>, framework::DatasetMode::ALL,
                combine(m_values,
                        n_values,
                        k_values,
                        b_values,
                        m0_values_precommit,
                        n0_values_precommit,
                        k0_values_precommit,
                        make("ExportToCLImage", false),
                        make("DataType", DataType::F32),
                        a_values,
                        beta_values,
                        broadcast_bias_values,
                        act_values))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedOnlyRhsMMULFixture<half>, framework::DatasetMode::ALL,
                combine(m_values,
                        n_values_fp16,
                        k_values_fp16,
                        b_values,
                        m0_values_precommit_fp16,
                        n0_values_precommit_fp16,
                        k0_values_precommit,
                        make("ExportToCLImage", false),
                        make("DataType", DataType::F16),
                        a_values,
                        beta_values,
                        broadcast_bias_values,
                        act_values))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16

TEST_SUITE(ExportToCLImage)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedOnlyRhsMMULFixture<float>, framework::DatasetMode::ALL,
                combine(m_values,
                        n_values,
                        k_values,
                        b_values,
                        m0_values_precommit,
                        n0_values_precommit,
                        k0_values_precommit,
                        make("ExportToCLImage", true),
                        make("DataType", DataType::F32),
                        a_values,
                        beta_values,
                        broadcast_bias_values,
                        act_values))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedOnlyRhsMMULFixture<half>, framework::DatasetMode::ALL,
                combine(m_values,
                        n_values_fp16,
                        k_values_fp16,
                        b_values,
                        m0_values_precommit_fp16,
                        n0_values_precommit_texture_fp16,
                        k0_values_precommit,
                        make("ExportToCLImage", true),
                        make("DataType", DataType::F16),
                        a_values,
                        beta_values,
                        broadcast_bias_values,
                        act_values))
{
    // Validate output
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // ExportToCLImage
TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMMMatrixMultiplyReshapedOnlyRhsMMUL
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
