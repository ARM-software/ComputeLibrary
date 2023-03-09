/*
 * Copyright (c) 2023 Arm Limited.
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

#include "arm_compute/runtime/CL/CLTensor.h"
#include "src/gpu/cl/kernels/ClNativeMatMulKernel.h"
#include "tests/datasets/LargeBatchMatMulDataset.h"
#include "tests/datasets/SmallBatchMatMulDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BatchMatMulFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float          abs_tolerance_f32(
    0.0001f); /**< Absolute tolerance value for comparing reference's output against implementation's output for floating point data types in case using relative tolerance fails because of small values */
constexpr float abs_tolerance_f16(
    0.001f);                                                   /**< Absolute tolerance value for comparing reference's output against implementation's output for fp16  data types in case using relative tolerance fails because of small values */
RelativeTolerance<half_float::half> tolerance_f16(half(0.01)); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
} // namespace

/** M0 values to test --precommit*/
const auto m0_values_precommit = framework::dataset::make("M0", { 1, 3 });

/** N0 values to test --precommit*/
const auto n0_values_precommit = framework::dataset::make("N0", { 2, 4 });

/** K0 values to test --precommit*/
const auto k0_values_precommit = framework::dataset::make("K0", { 2, 3 });

/** M0 values to test --nightly*/
const auto m0_values_nightly_lhs_nt = framework::dataset::make("M0", { 1, 2, 3, 4, 5, 6, 7, 8 });
// const auto m0_values_nightly_lhs_t = framework::dataset::make("M0", { 1, 2, 3, 4, 8 }); // To be enabled

/** N0 values to test --nightly*/
const auto n0_values_nightly_rhs_nt = framework::dataset::make("N0", { 1, 2, 3, 4, 8, 16 });
const auto n0_values_nightly_rhs_t  = framework::dataset::make("N0", { 1, 2, 3, 4, 8 });

/** K0 values to test --nightly*/
const auto k0_values_nightly_lhs_nt_rhs_nt = framework::dataset::make("K0", { 1, 2, 3, 4, 8, 16 });
const auto k0_values_nightly_lhs_nt_rhs_t  = framework::dataset::make("K0", { 1, 2, 3, 4, 8 });
// const auto k0_values_nightly_lhs_t_rhs_nt = framework::dataset::make("K0", { 1, 2, 3, 4, 5, 6, 7, 8 }); // To be enabled

template <typename T>
using CLBatchMatMulFixture = BatchMatMulValidationFixture<T>;

TEST_SUITE(CL)
TEST_SUITE(BatchMatMul)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
                                                                      framework::dataset::make("LhsInfo",
{
    TensorInfo(TensorShape(27U, 13U), 1, DataType::S32), // Unsupported data type
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
    TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
}),
framework::dataset::make("RhsInfo",
{
    TensorInfo(TensorShape(8U, 27U), 1, DataType::S32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32), TensorInfo(TensorShape(8U, 27U), 1, DataType::F32),
})),
framework::dataset::make("OutputInfo",
{
    TensorInfo(TensorShape(8U, 13U), 1, DataType::S32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32), TensorInfo(TensorShape(8U, 13U), 1, DataType::F32),
})),
framework::dataset::make("MatMulInfo",
{
    MatMulKernelInfo(false, false, 2, 2, 2, false), MatMulKernelInfo(false, false, 2, 2, 2, false), MatMulKernelInfo(false, false, 9, 2, 2, false), MatMulKernelInfo(false, false, 0, 2, 2, false), // M0 cannot be < 1
    MatMulKernelInfo(false, true, 4, 5, 2, false),                                                                                                                                                  // For LHS NT RHS NT: N0 cannot be 5
    MatMulKernelInfo(false, true, 4, 6, 2, false),                                                                                                                                                  // For LHS NT RHS NT: N0 cannot be 6
    MatMulKernelInfo(false, true, 4, 9, 2, false),                                                                                                                                                  // For LHS NT RHS NT: N0 cannot be 9
    MatMulKernelInfo(false, true, 4, 10, 2, false),                                                                                                                                                 // For LHS NT RHS NT: N0 cannot be 10
    MatMulKernelInfo(false, true, 4, 11, 2, false),                                                                                                                                                 // For LHS NT RHS NT: N0 cannot be 11
    MatMulKernelInfo(false, true, 4, 17, 2, false),                                                                                                                                                 // For LHS NT RHS NT: N0 cannot be 17
})),
framework::dataset::make("Expected", { false, true, true, false, false, false, false, false, false, false })),
lhs_info, rhs_info, output_info, matmul_info, expected)
{
    bool is_valid = bool(ClNativeMatMulKernel::validate(&lhs_info, &rhs_info, &output_info, matmul_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmallNoTranspose, CLBatchMatMulFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::SmallBatchMatMulDataset(),
                                                                                                                      framework::dataset::make("pretransose_A", { false })),
                                                                                                                      framework::dataset::make("pretransose_B", { false })),
                                                                                                                      m0_values_precommit),
                                                                                                                      n0_values_precommit),
                                                                                                                      k0_values_precommit),
                                                                                                              framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallRhsTransposed, CLBatchMatMulFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::SmallBatchMatMulDataset(),
                                                                                                                        framework::dataset::make("pretransose_A", { false })),
                                                                                                                        framework::dataset::make("pretransose_B", { true })),
                                                                                                                        m0_values_precommit),
                                                                                                                        n0_values_precommit),
                                                                                                                        k0_values_precommit),
                                                                                                                framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLargeNoTranspose, CLBatchMatMulFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(datasets::LargeBatchMatMulDataset(),
                                                                                                                  framework::dataset::make("pretransose_A", { false })),
                                                                                                                  framework::dataset::make("pretransose_B", { false })),
                                                                                                                  m0_values_nightly_lhs_nt),
                                                                                                                  n0_values_nightly_rhs_nt),
                                                                                                                  k0_values_nightly_lhs_nt_rhs_nt),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
// Running High Dimensional test is enough for FP32, because we're stressing the number of dimensions, not data type or M0/N0/K0
// It's a good idea to test for each Lhs/Rhs T/NT combinations because they're different CL kernels
FIXTURE_DATA_TEST_CASE(RunHighDimNoTranspose, CLBatchMatMulFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::HighDimensionalBatchMatMulDataset(),
                                                                                                                        framework::dataset::make("pretransose_A", { false })),
                                                                                                                        framework::dataset::make("pretransose_B", { false })),
                                                                                                                        framework::dataset::make("M0", { 2 })),
                                                                                                                        framework::dataset::make("N0", { 2 })),
                                                                                                                        framework::dataset::make("K0", { 2 })),
                                                                                                                framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLargeRhsTransposed, CLBatchMatMulFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(datasets::LargeBatchMatMulDataset(),
                                                                                                                    framework::dataset::make("pretransose_A", { false })),
                                                                                                                    framework::dataset::make("pretransose_B", { true })),
                                                                                                                    m0_values_nightly_lhs_nt),
                                                                                                                    n0_values_nightly_rhs_t),
                                                                                                                    k0_values_nightly_lhs_nt_rhs_t),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunHighDimRhsTransposed, CLBatchMatMulFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::HighDimensionalBatchMatMulDataset(),
                                                                                                                  framework::dataset::make("pretransose_A", { false })),
                                                                                                                  framework::dataset::make("pretransose_B", { true })),
                                                                                                                  framework::dataset::make("M0", { 2 })),
                                                                                                                  framework::dataset::make("N0", { 2 })),
                                                                                                                  framework::dataset::make("K0", { 2 })),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmallNoTranspose, CLBatchMatMulFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::SmallBatchMatMulDataset(),
                                                                                                                     framework::dataset::make("pretransose_A", { false })),
                                                                                                                     framework::dataset::make("pretransose_B", { false })),
                                                                                                                     m0_values_precommit),
                                                                                                                     n0_values_precommit),
                                                                                                                     k0_values_precommit),
                                                                                                             framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmallRhsTransposed, CLBatchMatMulFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::SmallBatchMatMulDataset(),
                                                                                                                       framework::dataset::make("pretransose_A", { false })),
                                                                                                                       framework::dataset::make("pretransose_B", { true })),
                                                                                                                       m0_values_precommit),
                                                                                                                       n0_values_precommit),
                                                                                                                       k0_values_precommit),
                                                                                                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLargeNoTranspose, CLBatchMatMulFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(datasets::LargeBatchMatMulDataset(),
                                                                                                                 framework::dataset::make("pretransose_A", { false })),
                                                                                                                 framework::dataset::make("pretransose_B", { false })),
                                                                                                                 m0_values_nightly_lhs_nt),
                                                                                                                 n0_values_nightly_rhs_nt),
                                                                                                                 k0_values_nightly_lhs_nt_rhs_nt),
                                                                                                                 framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLargeRhsTransposed, CLBatchMatMulFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(datasets::LargeBatchMatMulDataset(),
                                                                                                                   framework::dataset::make("pretransose_A", { false })),
                                                                                                                   framework::dataset::make("pretransose_B", { true })),
                                                                                                                   m0_values_nightly_lhs_nt),
                                                                                                                   n0_values_nightly_rhs_t),
                                                                                                                   k0_values_nightly_lhs_nt_rhs_t),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16

TEST_SUITE_END() // Float
TEST_SUITE_END() // BatchMatMul
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
