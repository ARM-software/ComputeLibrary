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
#include "FixedPointTarget.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FixedPointFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_exp_qs8(0.0f);          /**< Tolerance value for comparing reference's output against implementation's output  (exponential) for DataType::QS8 */
constexpr AbsoluteTolerance<float> tolerance_exp_qs16(1.0f);         /**< Tolerance value for comparing reference's output against implementation's output  (exponential) for DataType::QS16 */
constexpr AbsoluteTolerance<float> tolerance_invsqrt_qs8(4.0f);      /**< Tolerance value for comparing reference's output against implementation's output (inverse square-root) for DataType::QS8 */
constexpr AbsoluteTolerance<float> tolerance_invsqrt_qs16(5.0f);     /**< Tolerance value for comparing reference's output against implementation's output (inverse square-root) for DataType::QS16 */
constexpr AbsoluteTolerance<float> tolerance_log_qs8(5.0f);          /**< Tolerance value for comparing reference's output against implementation's output (logarithm) for DataType::QS8 */
constexpr AbsoluteTolerance<float> tolerance_log_qs16(7.0f);         /**< Tolerance value for comparing reference's output against implementation's output (logarithm) for DataType::QS16 */
constexpr AbsoluteTolerance<float> tolerance_reciprocal_qs8(3);      /**< Tolerance value for comparing reference's output against implementation's output (reciprocal) for DataType::QS8 */
constexpr AbsoluteTolerance<float> tolerance_reciprocal_qs16(11.0f); /**< Tolerance value for comparing reference's output against implementation's output (reciprocal) for DataType::QS16 */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(FixedPoint)
template <typename T>
using NEFixedPointFixture = FixedPointValidationFixture<Tensor, Accessor, T>;

TEST_SUITE(QS8)
TEST_SUITE(Exp)

FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                           framework::dataset::make("FixedPointOp", FixedPointOp::EXP)),
                                                                                                   framework::dataset::make("FractionalBits", 1, 7)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_exp_qs8, 0);
}
TEST_SUITE_END()

TEST_SUITE(Invsqrt)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                           framework::dataset::make("FixedPointOp", FixedPointOp::INV_SQRT)),
                                                                                                   framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_invsqrt_qs8, 0);
}
TEST_SUITE_END()

TEST_SUITE(Log)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                           framework::dataset::make("FixedPointOp", FixedPointOp::LOG)),
                                                                                                   framework::dataset::make("FractionalBits", 3, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_log_qs8, 0);
}
TEST_SUITE_END()

TEST_SUITE(Reciprocal)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                           framework::dataset::make("FixedPointOp", FixedPointOp::RECIPROCAL)),
                                                                                                   framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_reciprocal_qs8, 0);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(QS16)
TEST_SUITE(Exp)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                    DataType::QS16)),
                                                                                                            framework::dataset::make("FixedPointOp", FixedPointOp::EXP)),
                                                                                                    framework::dataset::make("FractionalBits", 1, 15)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_exp_qs16, 0);
}
TEST_SUITE_END()

TEST_SUITE(Invsqrt)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::make("Shape", TensorShape(8192U)),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::QS16)),
                                                                                                            framework::dataset::make("FixedPointOp", FixedPointOp::INV_SQRT)),
                                                                                                    framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_invsqrt_qs16, 0);
}
TEST_SUITE_END()

TEST_SUITE(Log)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                    DataType::QS16)),
                                                                                                            framework::dataset::make("FixedPointOp", FixedPointOp::LOG)),
                                                                                                    framework::dataset::make("FractionalBits", 4, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_log_qs16, 0);
}
TEST_SUITE_END()

TEST_SUITE(Reciprocal)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                    DataType::QS16)),
                                                                                                            framework::dataset::make("FixedPointOp", FixedPointOp::RECIPROCAL)),
                                                                                                    framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_reciprocal_qs16, 0);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
