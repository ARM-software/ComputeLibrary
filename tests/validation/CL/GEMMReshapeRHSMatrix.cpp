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
#include "tests/validation/fixtures/GEMMReshapeRHSMatrixFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// *INDENT-OFF*
// clang-format off
/** Data types */
const auto data_types = framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F16, DataType::F32 });

/** Batch size values to test */
const auto b_values = framework::dataset::make("batchsize", 1, 3);

/** N0 values to test - Precommit */
const auto n0_values_precommit = framework::dataset::make("N0", { 2, 4 });

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 2, 3, 4, 8, 16 });

/** K0 values to test (transpose=true) - Precommit */
const auto k0_t_values_precommit = framework::dataset::make("K0", { 4 });

/** K0 values to test (transpose=true) - Nightly */
const auto k0_t_values_nightly = framework::dataset::make("K0", { 2, 3, 4, 8, 16 });

/** K0 values to test (transpose=false) - Precommit */
const auto k0_nt_values_precommit = framework::dataset::make("K0", { 1, 2, 4 });

/** K0 values to test (transpose=false) - Nightly */
const auto k0_nt_values_nightly = framework::dataset::make("K0", { 1, 2, 3, 4, 8, 16 });

/** H0 values to test */
const auto h0_values = framework::dataset::make("H0", 1, 4);

/** Interleave values to test */
const auto i_values = framework::dataset::make("interleave", { true, false });

} // namespace

using namespace arm_compute::misc::shape_calculator;

// Initialize the output tensor with zero and fill the border with zero
using CLGEMMReshapeRHSMatrix = CLSynthetizeFunctionInitOutputWithZeroAndWithZeroConstantBorder<CLGEMMReshapeRHSMatrixKernel, 16>;

template <typename T>
using CLGEMMReshapeRHSMatrixFixture = GEMMReshapeRHSMatrixValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeRHSMatrix, T>;

TEST_SUITE(CL)
TEST_SUITE(GEMMReshapeRHSMatrix)

// This configuration tests only transpose = true
DATA_TEST_CASE(Configuration0, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   data_types),
                                                                   n0_values_nightly),
                                                                   k0_t_values_nightly),
                                                                   h0_values),
                                                                   i_values),
shape_in, b_value, data_type, n0_value, k0_value, h0_value, i_value)
{
    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = n0_value;
    rhs_info.k0         = k0_value;
    rhs_info.h0         = h0_value;
    rhs_info.interleave = i_value;
    rhs_info.transpose  = true;

    const TensorShape shape_src(shape_in[0], shape_in[1], b_value);
    const TensorShape shape_dst = compute_rhs_reshaped_shape(TensorInfo(shape_src, 1, data_type), rhs_info);

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape_src, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape_dst, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMReshapeRHSMatrixKernel reshape_rhs;
    reshape_rhs.configure(&src, &dst, rhs_info);
}

// This configuration tests only transpose = false
DATA_TEST_CASE(Configuration1, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   data_types),
                                                                   n0_values_nightly),
                                                                   k0_nt_values_nightly),
                                                                   h0_values),
                                                                   i_values),
shape_in, b_value, data_type, n0_value, k0_value, h0_value, i_value)
{
    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = n0_value;
    rhs_info.k0         = k0_value;
    rhs_info.h0         = h0_value;
    rhs_info.interleave = i_value;
    rhs_info.transpose  = false;

    const TensorShape shape_src(shape_in[0], shape_in[1], b_value);
    const TensorShape shape_dst = compute_rhs_reshaped_shape(TensorInfo(shape_src, 1, data_type), rhs_info);

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape_src, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape_dst, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMReshapeRHSMatrixKernel reshape_rhs;
    reshape_rhs.configure(&src, &dst, rhs_info);
}

TEST_SUITE(S32)
// RunSmall tests only for transpose = false
FIXTURE_DATA_TEST_CASE(RunSmall0, CLGEMMReshapeRHSMatrixFixture<int>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   n0_values_precommit),
                                                                   k0_nt_values_precommit),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunSmall tests only for transpose = true
FIXTURE_DATA_TEST_CASE(RunSmall1, CLGEMMReshapeRHSMatrixFixture<int>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   n0_values_precommit),
                                                                   k0_t_values_precommit),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunLarge tests only for transpose = false
FIXTURE_DATA_TEST_CASE(RunLarge0, CLGEMMReshapeRHSMatrixFixture<int>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   n0_values_nightly),
                                                                   k0_nt_values_nightly),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunLarge tests only for transpose = true
FIXTURE_DATA_TEST_CASE(RunLarge1, CLGEMMReshapeRHSMatrixFixture<int>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   n0_values_nightly),
                                                                   k0_t_values_nightly),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
// RunSmall tests only for transpose = false
FIXTURE_DATA_TEST_CASE(RunSmall0, CLGEMMReshapeRHSMatrixFixture<short>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   n0_values_precommit),
                                                                   k0_nt_values_precommit),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunSmall tests only for transpose = true
FIXTURE_DATA_TEST_CASE(RunSmall1, CLGEMMReshapeRHSMatrixFixture<short>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   n0_values_precommit),
                                                                   k0_t_values_precommit),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunLarge tests only for transpose = false
FIXTURE_DATA_TEST_CASE(RunLarge0, CLGEMMReshapeRHSMatrixFixture<short>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   n0_values_nightly),
                                                                   k0_nt_values_nightly),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunLarge tests only for transpose = true
FIXTURE_DATA_TEST_CASE(RunLarge1, CLGEMMReshapeRHSMatrixFixture<short>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   n0_values_nightly),
                                                                   k0_t_values_nightly),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
// RunSmall tests only for transpose = false
FIXTURE_DATA_TEST_CASE(RunSmall0, CLGEMMReshapeRHSMatrixFixture<char>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   n0_values_precommit),
                                                                   k0_nt_values_precommit),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunSmall tests only for transpose = true
FIXTURE_DATA_TEST_CASE(RunSmall1, CLGEMMReshapeRHSMatrixFixture<char>, framework::DatasetMode::PRECOMMIT,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   n0_values_precommit),
                                                                   k0_t_values_precommit),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunLarge tests only for transpose = false
FIXTURE_DATA_TEST_CASE(RunLarge0, CLGEMMReshapeRHSMatrixFixture<char>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   n0_values_nightly),
                                                                   k0_nt_values_nightly),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// RunLarge tests only for transpose = true
FIXTURE_DATA_TEST_CASE(RunLarge1, CLGEMMReshapeRHSMatrixFixture<char>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   n0_values_nightly),
                                                                   k0_t_values_nightly),
                                                                   h0_values),
                                                                   i_values),
                                                                   framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // GEMMReshapeRHSMatrix
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute