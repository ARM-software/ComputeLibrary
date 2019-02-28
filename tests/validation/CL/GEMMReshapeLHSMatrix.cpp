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
const auto data_types = framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F16, DataType::F32 });

/** Batch size values to test */
const auto b_values = framework::dataset::make("batchsize", 1, 3);

/** M0 values to test - Precommit */
const auto m0_values_precommit = framework::dataset::make("M0", { 2, 4, 5 });

/** K0 values to test - Precommit */
const auto k0_values_precommit = framework::dataset::make("K0", { 2, 4 });

/** M0 values to test - Precommit */
const auto m0_values_nightly = framework::dataset::make("M0", 2, 9);

/** K0 values to test - Precommit */
const auto k0_values_nightly = framework::dataset::make("K0", { 2, 3, 4, 8, 16 });

/** V0 values to test */
const auto v0_values = framework::dataset::make("V0", 1, 4);

/** Interleave values to test */
const auto i_values = framework::dataset::make("interleave", { true, false });

/** Transpose values to test */
const auto t_values = framework::dataset::make("transpose", { true, false });

/** Configuration test */
void validate_configuration(TensorShape shape_in, unsigned int b_value, DataType data_type, unsigned int m0_value, unsigned int k0_value, unsigned int v0_value, bool i_value, bool t_value, bool reinterpret_input_as_3d)
{
    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = m0_value;
    lhs_info.k0         = k0_value;
    lhs_info.v0         = v0_value;
    lhs_info.interleave = i_value;
    lhs_info.transpose  = t_value;

    TensorShape shape_src = shape_in;
    shape_src.set(shape_src.num_dimensions(), b_value);

    const TensorShape shape_dst = compute_lhs_reshaped_shape(TensorInfo(shape_src, 1, data_type), lhs_info, reinterpret_input_as_3d);

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape_src, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape_dst, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMReshapeLHSMatrixKernel reshape_lhs;
    reshape_lhs.configure(&src, &dst, lhs_info, reinterpret_input_as_3d);
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMReshapeLHSMatrix)

DATA_TEST_CASE(ConfigurationSmall, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   data_types),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values),
shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value)
{
    validate_configuration(shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value, false);
}

DATA_TEST_CASE(ConfigurationLarge, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   data_types),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values),
shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value)
{
    validate_configuration(shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value, false);
}

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMReshapeLHSMatrixFixture<int>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMReshapeLHSMatrixFixture<int>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMReshapeLHSMatrixFixture<short>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMReshapeLHSMatrixFixture<short>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMReshapeLHSMatrixFixture<char>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMReshapeLHSMatrixFixture<char>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape2DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S8

TEST_SUITE(ReinterpretInputAs3D)
DATA_TEST_CASE(ConfigurationSmall, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   data_types),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values),
shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value)
{
    validate_configuration(shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value, true);
}

DATA_TEST_CASE(ConfigurationLarge, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   data_types),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values),
shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value)
{
    validate_configuration(shape_in, b_value, data_type, m0_value, k0_value, v0_value, i_value, t_value, true);
}

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMReshapeLHSMatrix3DFixture<int>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMReshapeLHSMatrix3DFixture<int>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S32)),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMReshapeLHSMatrix3DFixture<short>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMReshapeLHSMatrix3DFixture<short>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S16)),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMReshapeLHSMatrix3DFixture<char>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   m0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMReshapeLHSMatrix3DFixture<char>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(datasets::LargeGEMMReshape3DShapes(),
                                                                   b_values),
                                                                   framework::dataset::make("DataType", DataType::S8)),
                                                                   m0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values),
                                                                   i_values),
                                                                   t_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // ReinterpretInputAs3D
TEST_SUITE_END() // GEMMReshapeLHSMatrix
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute