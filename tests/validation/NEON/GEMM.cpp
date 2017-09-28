/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeGEMMDataset.h"
#include "tests/datasets/SmallGEMMDataset.h"
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
constexpr AbsoluteTolerance<float> tolerance_f(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr AbsoluteTolerance<float> tolerance_q(1.0f);   /**< Tolerance value for comparing reference's output against implementation's output for fixed point data types */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef ARM_COMPUTE_ENABLE_FP16
    DataType::F16,
#endif /* ARM_COMPUTE_ENABLE_FP16 */
    DataType::F32,
    DataType::QS8,
    DataType::QS16,
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(GEMM)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallGEMMDataset(), datasets::LargeGEMMDataset()), CNNDataTypes),
               shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type)
{
    // Set fixed point position data type allowed
    const int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    // Create tensors
    Tensor a   = create_tensor<Tensor>(shape_a, data_type, 1, fixed_point_position);
    Tensor b   = create_tensor<Tensor>(shape_b, data_type, 1, fixed_point_position);
    Tensor c   = create_tensor<Tensor>(shape_c, data_type, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(output_shape, data_type, 1, fixed_point_position);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEGEMM gemm;
    gemm.configure(&a, &b, &c, &dst, alpha, beta);
}

template <typename T>
using NEGEMMFixture = GEMMValidationFixture<Tensor, Accessor, NEGEMM, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMDataset(), framework::dataset::make("DataType",
                                                                                               DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NEGEMMFixedPointFixture = GEMMValidationFixedPointFixture<Tensor, Accessor, NEGEMM, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::QS8)),
                                                                                                             framework::dataset::make("FractionalBits", 1, 7)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                                   framework::dataset::make("DataType",
                                                                                                                           DataType::QS8)),
                                                                                                           framework::dataset::make("FractionalBits", 1, 7)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::QS16)),
                                                                                                              framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::QS16)),
                                                                                                            framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
