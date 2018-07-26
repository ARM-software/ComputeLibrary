/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
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
#include "tests/validation/fixtures/GEMMInterleave4x4Fixture.h"
#include "tests/validation/fixtures/GEMMTranspose1xWFixture.h"

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
constexpr float                     tolerance_num   = 0.02f;  /**< Tolerance number */
const auto                          data_interleave = framework::dataset::make("M", 8, 14) * framework::dataset::make("N", 7, 14);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
});
} // namespace

const auto data_transpose = framework::dataset::make("M", 8, 14) * framework::dataset::make("N", 7, 14);

TEST_SUITE(CL)
TEST_SUITE(GEMM)

TEST_SUITE(INTERLEAVE_4X4)
using CLGEMMInterleave4x4 = CLSynthetizeFunctionWithZeroConstantBorder<CLGEMMInterleave4x4Kernel, 4>;

TEST_SUITE(FP32)
using CLGEMMInterleave4x4Fixture = GEMMInterleave4x4ValidationFixture<CLTensor, CLAccessor, CLGEMMInterleave4x4, float>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMInterleave4x4Fixture, framework::DatasetMode::PRECOMMIT, data_interleave * framework::dataset::make("DataType", DataType::F32))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32

TEST_SUITE_END() // INTERLEAVE_4X4

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallGEMMDataset(), datasets::LargeGEMMDataset()), CNNDataTypes),
               shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type)
{
    // Create tensors
    CLTensor a   = create_tensor<CLTensor>(shape_a, data_type, 1);
    CLTensor b   = create_tensor<CLTensor>(shape_b, data_type, 1);
    CLTensor c   = create_tensor<CLTensor>(shape_c, data_type, 1);
    CLTensor dst = create_tensor<CLTensor>(output_shape, data_type, 1);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMM gemm;
    gemm.configure(&a, &b, &c, &dst, alpha, beta);

    //TODO(COMPMID-415): Validate valid region
}

template <typename T>
using CLGEMMFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T>;

template <typename T>
using CLGEMMOutput3DFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T, false, true>;

template <typename T>
using CLGEMMInputOutput3DFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T, true, true>;

TEST_SUITE(TRANSPOSE_1XW)
using CLGEMMTranspose1xW        = CLSynthetizeFunctionWithZeroConstantBorder<CLGEMMTranspose1xWKernel, 4>;
using CLGEMMTranspose1xWFixture = GEMMTranspose1xWValidationFixture<CLTensor, CLAccessor, CLGEMMTranspose1xW, float>;
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMTranspose1xWFixture, framework::DatasetMode::PRECOMMIT, data_transpose * framework::dataset::make("DataType", DataType::F32))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32

TEST_SUITE_END() //TRANSPOSE_1XW

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMDataset(), framework::dataset::make("DataType",
                                                                                               DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(INPUT_OUTPUT_3D)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMInputOutput3DFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMInputOutput3DDataset(),
                                                                                                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMInputOutput3DFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMInputOutput3DDataset(),
                                                                                                             framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMInputOutput3DFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMInputOutput3DDataset(),
                                                                                                              framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMInputOutput3DFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMInputOutput3DDataset(),
                                                                                                            framework::dataset::make("DataType",
                                                                                                                    DataType::F16)))
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
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMOutput3DFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMOutput3DDataset(),
                                                                                                          framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMOutput3DFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMOutput3DDataset(),
                                                                                                        framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMOutput3DFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMOutput3DDataset(),
                                                                                                         framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMOutput3DFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMOutput3DDataset(),
                                                                                                       framework::dataset::make("DataType",
                                                                                                               DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // FP16

TEST_SUITE_END() // Float
TEST_SUITE_END() // OUTPUT_3D

TEST_SUITE_END() // GEMM
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
