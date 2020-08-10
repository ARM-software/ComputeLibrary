/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMReshapeLHSMatrixKernel.h"
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
#include "tests/validation/fixtures/GEMMReshapeLHSMatrixFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

// Initialize the output tensor with zero and fill the border with zero
using CLGEMMReshapeLHSMatrix = CLSynthetizeFunctionInitOutputWithZeroAndWithZeroConstantBorder<CLGEMMReshapeLHSMatrixKernel, 16>;

template <typename T>
using CLGEMMReshapeLHSMatrixFixture = GEMMReshapeLHSMatrixValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeLHSMatrix, T, false>;

// Fixture to use when the input has to be reinterpreted as 3D
template <typename T>
using CLGEMMReshapeLHSMatrix3DFixture = GEMMReshapeLHSMatrixValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeLHSMatrix, T, true>;

// *INDENT-OFF*
// clang-format off
/** Data types */

namespace
{
/** Batch size values to test */
const auto b_values = framework::dataset::make("batchsize", 1, 3);

/** M0 values to test */
const auto m0_values_s32 = framework::dataset::make("M0", { 2, 3 });
const auto m0_values_s16 = framework::dataset::make("M0", { 4, 5 });
const auto m0_values_s8 = framework::dataset::make("M0", { 6, 7, 8 });

/** K0 values to test */
const auto k0_values_s32 = framework::dataset::make("K0", { 2, 3 });
const auto k0_values_s16 = framework::dataset::make("K0", { 4, 8 });
const auto k0_values_s8 = framework::dataset::make("K0", { 16 });

/** V0 values to test */
const auto v0_values = framework::dataset::make("V0", 1, 4);

/** Interleave values to test */
const auto i_values = framework::dataset::make("interleave", { true, false });

/** Transpose values to test */
const auto t_values = framework::dataset::make("transpose", { true, false });

/** Zero padding test */
bool validate_zero_padding(unsigned int m_value, unsigned int k_value, unsigned int b_value, unsigned int m0_value, unsigned int k0_value, unsigned int v0_value,
                            bool i_value_lhs, bool t_value_lhs, bool input_as_3d, DataType dt)
{
    const unsigned int M = m_value;
    const unsigned int K = k_value;
    const unsigned int B = b_value;

    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0 = m0_value;
    lhs_info.k0 = k0_value;
    lhs_info.v0 = v0_value;
    lhs_info.interleave = i_value_lhs;
    lhs_info.transpose = t_value_lhs;

    const TensorShape lhs_shape(K, M, B);
    const TensorShape lhs_shape_reshaped = compute_lhs_reshaped_shape(TensorInfo(lhs_shape, 1, dt), lhs_info, input_as_3d);

    // Create tensors
    CLTensor lhs = create_tensor<CLTensor>(lhs_shape, dt);
    CLTensor dst = create_tensor<CLTensor>(lhs_shape_reshaped, dt);

    ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Validate zero-padding
    CLGEMMReshapeLHSMatrixKernel lhs_reshape;

    lhs_reshape.configure(&lhs, &dst, lhs_info, input_as_3d);

    return lhs.info()->padding().empty();
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMReshapeLHSMatrix)

/** Validate zero padding tests for the LHS input tensor
 *
 * A series of validation tests to test the zero padding requirement
 *
 * Checks performed in order:
 *     - Case where M and K are smaller than M0 and K0
 *     - Generic test case with batch size = 1
 *     - Generic test case with batch size = 4
 *     - Generic test case with input_as_3d_value = true
 */
DATA_TEST_CASE(ValidateZeroPadding, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
framework::dataset::make("M",                   { 1, 23, 63, 101 }),
framework::dataset::make("K",                   { 1, 47, 29,  27 })),
framework::dataset::make("B",                   { 1, 1, 4, 7 })),
framework::dataset::make("M0",                  { 4, 2, 4, 8 })),
framework::dataset::make("K0",                  { 2, 2, 4, 8 })),
framework::dataset::make("input_as_3d",         { false, false, false, true })),
m_value, k_value, b_value, m0_value, k0_value, input_as_3d_value)
{
    constexpr DataType dt = DataType::F32;

    bool status = validate_zero_padding(m_value, k_value, b_value, m0_value, k0_value, 2, false, false, input_as_3d_value, dt);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(S32, CLGEMMReshapeLHSMatrixFixture<int>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   m0_values_s32),
                                                                   k0_values_s32),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(S16, CLGEMMReshapeLHSMatrixFixture<short>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   m0_values_s16),
                                                                   k0_values_s16),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(S8, CLGEMMReshapeLHSMatrixFixture<char>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   m0_values_s8),
                                                                   k0_values_s8),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE(ReinterpretInputAs3D)
FIXTURE_DATA_TEST_CASE(S32, CLGEMMReshapeLHSMatrix3DFixture<int>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   m0_values_s32),
                                                                   k0_values_s32),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(S16, CLGEMMReshapeLHSMatrix3DFixture<short>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   m0_values_s16),
                                                                   k0_values_s16),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(S8, CLGEMMReshapeLHSMatrix3DFixture<char>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   m0_values_s8),
                                                                   k0_values_s8),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // ReinterpretInputAs3D
TEST_SUITE_END() // GEMMReshapeLHSMatrix
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
