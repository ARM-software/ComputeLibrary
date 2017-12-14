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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "tests/CL/CLAccessor.h"
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
constexpr AbsoluteTolerance<float> tolerance_exp(1.0f);     /**< Tolerance value for comparing reference's output against implementation's output  (exponential)*/
constexpr AbsoluteTolerance<float> tolerance_invsqrt(4.0f); /**< Tolerance value for comparing reference's output against implementation's output (inverse square-root) */
constexpr AbsoluteTolerance<float> tolerance_log(5.0f);     /**< Tolerance value for comparing reference's output against implementation's output (logarithm) */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(FixedPoint)
TEST_SUITE(QS8)

template <typename T>
using CLFixedPointFixture = FixedPointValidationFixture<CLTensor, CLAccessor, T>;

TEST_SUITE(Exp)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                           framework::dataset::make("FixedPointOp", FixedPointOp::EXP)),
                                                                                                   framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_exp);
}
TEST_SUITE_END()

TEST_SUITE(Log)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                           framework::dataset::make("FixedPointOp", FixedPointOp::LOG)),
                                                                                                   framework::dataset::make("FractionalBits", 3, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_log);
}
TEST_SUITE_END()

TEST_SUITE(Invsqrt)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small1DShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                           framework::dataset::make("FixedPointOp", FixedPointOp::INV_SQRT)),
                                                                                                   framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_invsqrt);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
