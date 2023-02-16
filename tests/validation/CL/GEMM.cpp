/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeGEMMDataset.h"
#include "tests/datasets/SmallGEMMDataset.h"
#include "tests/datasets/TinyGEMMDataset.h"
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
namespace
{
RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float          abs_tolerance_f32(
    0.0001f);                                                 /**< Absolute tolerance value for comparing reference's output against implementation's output for floating point data types in case using relative tolerance fails because of small values */
RelativeTolerance<half_float::half> tolerance_f16(half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float                     tolerance_num = 0.02f;    /**< Tolerance number */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMM)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("LhsInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::S32), // Unsupported data type
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
                                                     }),
               framework::dataset::make("RhsInfo",{ TensorInfo(TensorShape(8U, 27U), 1, DataType::S32),
                                                        TensorInfo(TensorShape(8U, 27U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(8U, 13U), 1, DataType::S32),
                                                        TensorInfo(TensorShape(8U, 13U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { false, true })),
               lhs_info, rhs_info, output_info, expected)
{
    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;
    const auto gemm_info = GEMMInfo();
    bool is_valid = bool(CLGEMM::validate(&lhs_info.clone()->set_is_resizable(true), &rhs_info.clone()->set_is_resizable(true), nullptr, &output_info.clone()->set_is_resizable(true), alpha, beta, gemm_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
template <typename T>
using CLGEMMFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T>;

template <typename T>
using CLGEMMOutput3DFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T, false, false, true>;

template <typename T>
using CLGEMMInputOutput3DFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T, false, true, true>;

template <typename T>
using CLBatchedMatMulFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T, true, false, false, false, false, true>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                         framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                 framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                          framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                  framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                        framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(INPUT_OUTPUT_3D)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMInputOutput3DFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMInputOutput3DDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMInputOutput3DFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMInputOutput3DDataset(),
                                                                                                                     framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                             framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMInputOutput3DFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMInputOutput3DDataset(),
                                                                                                                      framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                              framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMInputOutput3DFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMInputOutput3DDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                            framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // FP16

TEST_SUITE_END() // Float
TEST_SUITE_END() // INPUT_OUTPUT_3D

TEST_SUITE(OUTPUT_3D)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMOutput3DFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMOutput3DDataset(),
                                                                                                                  framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                          framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMOutput3DFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMOutput3DDataset(),
                                                                                                                framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                        framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMOutput3DFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMOutput3DDataset(),
                                                                                                                 framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                         framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMOutput3DFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMOutput3DDataset(),
                                                                                                               framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                       framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // OUTPUT_3D

TEST_SUITE(BATCHED_MATMUL)

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLBatchedMatMulFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallBatchedMatMulDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { false })),
                                                                                                           framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLBatchedMatMulFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallBatchedMatMulDataset(),
                                                                                                                  framework::dataset::make("ReshapeWeights", { false })),
                                                                                                          framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END()
TEST_SUITE_END() // BATCHED_MATMUL

TEST_SUITE_END() // GEMM
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
