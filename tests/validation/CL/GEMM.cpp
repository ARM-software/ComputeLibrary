/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
RelativeTolerance<float>            tolerance_f32(0.001f);    /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
RelativeTolerance<half_float::half> tolerance_f16(half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr AbsoluteTolerance<float>  tolerance_q(1.0f);        /**< Tolerance value for comparing reference's output against implementation's output for fixed point data types */
constexpr float                     tolerance_num   = 0.02f;  /**< Tolerance number */
const auto                          data_interleave = framework::dataset::make("M", 8, 14) * framework::dataset::make("N", 7, 14);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
    DataType::QS8,
    DataType::QS16,
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

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
using CLGEMMInterleave4x4Fixture = GEMMInterleave4x4ValidationFixedPointFixture<CLTensor, CLAccessor, CLGEMMInterleave4x4, int8_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMInterleave4x4Fixture, framework::DatasetMode::PRECOMMIT, data_interleave *
                       framework::dataset::make("DataType", DataType::QS8)
                       * framework::dataset::make("FractionalBits", 1, 7))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
using CLGEMMInterleave4x4Fixture = GEMMInterleave4x4ValidationFixedPointFixture<CLTensor, CLAccessor, CLGEMMInterleave4x4, int16_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMInterleave4x4Fixture, framework::DatasetMode::PRECOMMIT, data_interleave *
                       framework::dataset::make("DataType", DataType::QS16)
                       * framework::dataset::make("FractionalBits", 1, 14))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()

TEST_SUITE_END() // INTERLEAVE_4X4

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallGEMMDataset(), datasets::LargeGEMMDataset()), CNNDataTypes),
               shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type)
{
    // Set fixed point position data type allowed
    const int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    // Create tensors
    CLTensor a   = create_tensor<CLTensor>(shape_a, data_type, 1, fixed_point_position);
    CLTensor b   = create_tensor<CLTensor>(shape_b, data_type, 1, fixed_point_position);
    CLTensor c   = create_tensor<CLTensor>(shape_c, data_type, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(output_shape, data_type, 1, fixed_point_position);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMM gemm;
    gemm.configure(&a, &b, &c, &dst, alpha, beta);
}

template <typename T>
using CLGEMMFixture = GEMMValidationFixture<CLTensor, CLAccessor, CLGEMM, T>;

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

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
using CLGEMMTranspose1xW        = CLSynthetizeFunctionWithZeroConstantBorder<CLGEMMTranspose1xWKernel, 16>;
using CLGEMMTranspose1xWFixture = GEMMTranspose1xWValidationFixedPointFixture<CLTensor, CLAccessor, CLGEMMTranspose1xW, int8_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMTranspose1xWFixture, framework::DatasetMode::PRECOMMIT, data_transpose *
                       framework::dataset::make("DataType", DataType::QS8)
                       * framework::dataset::make("FractionalBits", 1, 7))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
using CLGEMMTranspose1xW        = CLSynthetizeFunctionWithZeroConstantBorder<CLGEMMTranspose1xWKernel, 8>;
using CLGEMMTranspose1xWFixture = GEMMTranspose1xWValidationFixedPointFixture<CLTensor, CLAccessor, CLGEMMTranspose1xW, int16_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMTranspose1xWFixture, framework::DatasetMode::PRECOMMIT, data_transpose *
                       framework::dataset::make("DataType", DataType::QS16)
                       * framework::dataset::make("FractionalBits", 1, 14))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()

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
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLGEMMFixedPointFixture = GEMMValidationFixedPointFixture<CLTensor, CLAccessor, CLGEMM, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::QS8)),
                                                                                                             framework::dataset::make("FractionalBits", 1, 7)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                                   framework::dataset::make("DataType",
                                                                                                                           DataType::QS8)),
                                                                                                           framework::dataset::make("FractionalBits", 1, 7)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::QS16)),
                                                                                                              framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::QS16)),
                                                                                                            framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
