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
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyNativeKernel.h"
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

// Create function for CLGEMMMatrixMultiplyNativeKernel
using CLGEMMLowpMatrixMultiplyNative = CLSynthetizeFunction<CLGEMMLowpMatrixMultiplyNativeKernel>;

// Fixture for CLGEMMLowpMatrixMultiplyNative
using CLGEMMLowpMatrixMultiplyNativeFixture = GEMMLowpMatrixMultiplyNativeValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyNative>;

// Fixture for CLGEMMMatrixMultiplyNative3D
using CLGEMMLowpMatrixMultiplyNative3DFixture = GEMMLowpMatrixMultiplyNative3DValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyNative>;

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

/** M0 values to test - Nightly */
const auto m0_values_nightly = framework::dataset::make("M0", 1, 2, 7);

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 1, 2, 3, 4, 8 });

/** K0 values to test - Nightly */
const auto k0_values_nightly = framework::dataset::make("K0", { 1, 2, 3, 4, 8, 16 });

/** Zero padding test */
bool validate_zero_padding(unsigned int m_value, unsigned int n_value, unsigned int k_value, unsigned int b_value, unsigned int m0_value, unsigned int n0_value, unsigned int k0_value, bool broadcast_bias, DataType data_type, const ActivationLayerInfo &act_info)
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
    const TensorShape dst_shape = compute_mm_shape(TensorInfo(lhs_shape, 1, data_type),
                                                   TensorInfo(rhs_shape, 1, data_type),
                                                   kernel_info);

    // Create tensors
    CLTensor lhs  = create_tensor<CLTensor>(lhs_shape, data_type);
    CLTensor rhs  = create_tensor<CLTensor>(rhs_shape, data_type);
    CLTensor dst  = create_tensor<CLTensor>(dst_shape, DataType::S32);

    ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMLowpMatrixMultiplyNative gemm;
    gemm.configure(&lhs, &rhs,  &dst, lhs_info, rhs_info, GEMMReshapeInfo(m_value, n_value, k_value));

    // Padding can be added along rhs and bias's X dimension
    return dst.info()->padding().empty() && lhs.info()->padding().empty() && rhs.info()->padding().empty();
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMLowpMatrixMultiplyNative)

/** Validate zero padding tests
 *
 * A series of validation tests to check that no padding is added as part of configuration for 4 different scenarios.
 *
 * Checks performed in order:
 *     - No partial blocks in both x and y dimensions
 *     - Partial blocks in x dimension
 *     - Partial blocks in y dimension
 *     - Partial blocks in both x and y dimensions
 *     - No blocks in both x and y dimensions, scalar store (N0==1)
 *     - Special case: partial_n0 == 5 (vstore1 should be invoked instead of vstore_partial_1)
 */
DATA_TEST_CASE(ValidateZeroPadding, framework::DatasetMode::ALL, zip(zip(zip(
framework::dataset::make("M",                   { 24, 63,   1, 51, 255, }),
framework::dataset::make("N",                   { 47, 29, 122, 20,  21, })),
framework::dataset::make("M0",                  { 4,   8,   2,  1,   8, })),
framework::dataset::make("N0",                  { 4,   4,   3,  1,   8, })),
m_value, n_value, m0_value, n0_value)
{
    bool status = validate_zero_padding(m_value, n_value, 23, 1, m0_value, n0_value, 4, false, DataType::QASYMM8, ActivationLayerInfo());
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}



FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyNativeFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(m_values,
                                                                n_values),
                                                                k_values),
                                                                b_values),
                                                                m0_values_precommit),
                                                                n0_values_precommit),
                                                                k0_values_precommit))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyNativeFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(m_values,
                                                                n_values),
                                                                k_values),
                                                                b_values),
                                                                m0_values_nightly),
                                                                n0_values_nightly),
                                                                k0_values_nightly))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMLowpMatrixMultiplyNative3DFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(m_w_values,
                                                                        m_h_values),
                                                                        n_values),
                                                                        k_values),
                                                                        b_values),
                                                                        m0_values_precommit),
                                                                        n0_values_precommit),
                                                                        k0_values_precommit))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMLowpMatrixMultiplyNative3DFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(m_w_values,
                                                                        m_h_values),
                                                                        n_values),
                                                                        k_values),
                                                                        b_values),
                                                                        m0_values_nightly),
                                                                        n0_values_nightly),
                                                                        k0_values_nightly))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // GEMMLowpMatrixMultiplyNative
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
