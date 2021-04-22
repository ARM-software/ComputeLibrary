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
#include "src/core/gpu/cl/kernels/ClGemmMatrixMultiplyKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmReshapeLhsMatrixKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
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

// Create function for ClGemmReshapeLhsMatrixKernel
using CLGEMMReshapeLHSMatrix = CLSynthetizeOperator<ClGemmReshapeLhsMatrixKernel>;

// Create function for ClGemmReshapeRhsMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeOperator<ClGemmReshapeRhsMatrixKernel>;

// Create function for ClGemmMatrixMultiplyKernel
using CLGEMMMatrixMultiplyReshaped = CLSynthetizeOperator<ClGemmMatrixMultiplyKernel>;

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

/** Alpha values to test */
const auto alpha_values = framework::dataset::make("alpha", {1.0f, -0.75f} );

/** Beta values to test */
const auto beta_values = framework::dataset::make("beta", {-0.35f, 0.0f} );

/** M, N combinations to test
 *  1: Special 1x1 case
 *  2: Special multples of processor size in both dimensions
 *  3: Non multiples of processor size in both dimensions
*/
const auto m_n_values = zip(
    framework::dataset::make("M", {1, 16, 37}),
    framework::dataset::make("N", {1, 16, 51})
    );

/** N values to test */
const auto n_values = framework::dataset::make("N", 51);

/** K values to test */
const auto k_values = framework::dataset::make("K", 23);

/** M_W values to test */
const auto m_w_values = framework::dataset::make("M_W", 5);

/** M_H values to test */
const auto m_h_values = framework::dataset::make("M_H", 7);

/** Batch size values to test */
const auto b_values = framework::dataset::make("batch_size", 1, 3);

/** Activation values to test */
const auto act_values = framework::dataset::make("Activation",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
});

/** V0 values to test */
const auto v0_values = framework::dataset::make("V0", 2);

/** H0 values to test */
const auto h0_values = framework::dataset::make("H0", 4);

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
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMMatrixMultiplyInterleavedTransposed)
TEST_CASE(Negative, framework::DatasetMode::ALL)
{
    // The following tests are already integrated in the GEMMMatrixMultiply validation because
    // in common with this validation
    // - Unsupported QASYMM8 data type
    // - Unsupported SIZE_T data type
    // - Mixed precision with F32
    // - Max number of dimensions LHS matrix
    // - Max number of dimensions RHS matrix

    // Invalid LHS dimensions
    {
        // The correct shape should be: lhs = TensorInfo(TensorShape(256U, 1U, 1U, 1U), 1, DataType::F32);
        const auto lhs                       = TensorInfo(TensorShape(256U, 2U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(104U, 3U, 1U, 1U), 1, DataType::F32);
        const auto bias                      = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = true;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(16, 24, 13, 2, 4, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = false;
        const auto status    = ClGemmMatrixMultiplyKernel::validate(&lhs, &rhs, &bias, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid RHS dimensions
    {
        const auto lhs                       = TensorInfo(TensorShape(256U, 1U, 1U, 1U), 1, DataType::F32);
        // The correct shape should be rhs = TensorInfo(TensorShape(104U, 3U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(104U, 4U, 1U, 1U), 1, DataType::F32);
        const auto bias                      = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = true;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(16, 24, 13, 2, 4, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = false;
        const auto status    = ClGemmMatrixMultiplyKernel::validate(&lhs, &rhs, &bias, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Broadcast bias
    {
        const auto lhs                       = TensorInfo(TensorShape(256U, 1U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(104U, 3U, 1U, 1U), 1, DataType::F32);
        // The correct shape should be bias = TensorInfo(TensorShape(24U, 1U, 1U, 1U), 1, DataType::F32);
        const auto bias                      = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = true;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(16, 24, 13, 2, 4, 0, false, true);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = false;
        const auto status    = ClGemmMatrixMultiplyKernel::validate(&lhs, &rhs, &bias, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid dimensions for the bias
    {
        const auto lhs                       = TensorInfo(TensorShape(256U, 1U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(104U, 3U, 1U, 1U), 1, DataType::F32);
        // The correct shape should be bias = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        const auto bias                      = TensorInfo(TensorShape(25U, 16U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = true;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(16, 24, 13, 2, 4, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = false;
        const auto status    = ClGemmMatrixMultiplyKernel::validate(&lhs, &rhs, &bias, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid dimensions for the output
    {
        const auto lhs                       = TensorInfo(TensorShape(256U, 1U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(104U, 3U, 1U, 1U), 1, DataType::F32);
        const auto bias                      = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        // The correct shape should be out = TensorInfo(TensorShape(24U, 16U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(24U, 13U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = true;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(16, 24, 13, 2, 4, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = false;
        const auto status    = ClGemmMatrixMultiplyKernel::validate(&lhs, &rhs, &bias, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }
}

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_n_values,
                                                                   k_values),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values),
                                                                   h0_values),
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
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values),
                                                                   h0_values),
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
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_n_values,
                                                                   k_values),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values),
                                                                   h0_values),
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
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   v0_values),
                                                                   h0_values),
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