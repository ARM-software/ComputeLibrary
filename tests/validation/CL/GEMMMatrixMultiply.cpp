/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
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

// Create function for CLGEMMMatrixMultiplyKernel
using CLGEMMMatrixMultiplyNative = CLSynthetizeFunction<CLGEMMMatrixMultiplyKernel>;

// Fixture for GEMMMatrixMultiplyValidationFixture
template <typename T>
using CLGEMMMatrixMultiplyNativeFixture = GEMMMatrixMultiplyValidationFixture<CLTensor, CLAccessor, T, CLGEMMMatrixMultiplyNative>;

// Fixture for GEMMMatrixMultiply3DValidationFixture
template <typename T>
using CLGEMMMatrixMultiplyNative3DFixture = GEMMMatrixMultiply3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMMatrixMultiplyNative>;

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
 *  4: Special 1x1003 case
*/
const auto m_n_values = zip(
    framework::dataset::make("M", {1, 16, 37, 1}),
    framework::dataset::make("N", {1, 16, 51, 1003})
    );

/** N values to test */
const auto n_values = framework::dataset::make("N", {51, 1003});

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

/** Broadcast bias from vector to matrix */
const auto broadcast_bias_values = framework::dataset::make("broadcast_bias", { false, true } );

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
TEST_SUITE(GEMMMatrixMultiply)
TEST_CASE(Negative, framework::DatasetMode::ALL)
{
    // Unsupported QASYMM8 data type
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 1U), 1, DataType::QASYMM8);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 1U), 1, DataType::QASYMM8);
        const auto out                       = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::QASYMM8);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, nullptr, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Unsupported SIZE_T data type
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 1U), 1, DataType::SIZET);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 1U), 1, DataType::SIZET);
        const auto out                       = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::SIZET);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, nullptr, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Mixed precision with F32
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info  = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = true;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, nullptr, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Max number of dimensions LHS matrix
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 1U, 4U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, nullptr, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Max number of dimensions RHS matrix
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 4U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 4U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(14U, 12U, 1U, 4U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, nullptr, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Broadcast bias
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 1U), 1, DataType::F16);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 1U), 1, DataType::F16);
        // The correct shape should be bias = TensorInfo(TensorShape(14U, 1U, 1U, 1U), 1, DataType::F32);
        const auto bias                      = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::F16);
        const auto out                       = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::F16);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, true);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = false;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, &bias, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid dimensions for the bias
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 1U), 1, DataType::F32);
        // The correct shape should be bias = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::F32);
        const auto bias                      = TensorInfo(TensorShape(14U, 8U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const bool fp_mixed_precision        = false;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, &bias, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target, fp_mixed_precision);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid dimensions for the output
    {
        const auto lhs                       = TensorInfo(TensorShape(13U, 12U, 1U, 1U), 1, DataType::F32);
        const auto rhs                       = TensorInfo(TensorShape(14U, 13U, 1U, 1U), 1, DataType::F32);
        // The correct shape should be out = TensorInfo(TensorShape(14U, 12U, 1U, 1U), 1, DataType::F32);
        const auto out                       = TensorInfo(TensorShape(14U, 7U, 1U, 1U), 1, DataType::F32);
        constexpr float alpha                = 1.3f;
        constexpr float beta                 = 0.7f;
        const bool is_interleaved_transposed = false;
        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(12, 14, 13, 1, 1, 0, false, false);
        const GPUTarget gpu_target           = GPUTarget::MIDGARD;
        const auto status    = CLGEMMMatrixMultiplyKernel::validate(&lhs, &rhs, nullptr, &out, alpha, beta, is_interleaved_transposed, reshape_info, gpu_target);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }
}

TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyNativeFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_n_values,
                                                                   k_values),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   framework::dataset::make("fp16_mixed_precision", false)),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   gpu_arch_values))
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
                                                                   alpha_values),
                                                                   beta_values),
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
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyNativeFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_n_values,
                                                                   k_values),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
                                                                   broadcast_bias_values),
                                                                   fp16_mixed_precision_values),
                                                                   act_values),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   gpu_arch_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyNative3DFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   alpha_values),
                                                                   beta_values),
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
TEST_SUITE_END() // GEMMMatrixMuliplty
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute