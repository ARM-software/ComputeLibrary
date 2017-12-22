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
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FixedPointPixelWiseMultiplicationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const float tolerance   = 1.f;
const float scale_255   = 1.f / 255.f;
const float scale_unity = 1.f;

// *INDENT-OFF*
// clang-format off
#define FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, SCALE, RP, FPP_START, FPP_END) \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, NEFixedPointPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                      \
                           combine(combine(combine(combine(combine(combine(                                                            \
                           datasets::SHAPES,                                                                                           \
                           framework::dataset::make("DataType1", DataType::DT1)),                                                      \
                           framework::dataset::make("DataType2", DataType::DT2)),                                                      \
                           framework::dataset::make("Scale", std::move(SCALE))),                                                       \
                           datasets::ConvertPolicies()),                                                                               \
                           framework::dataset::make("RoundingPolicy", RoundingPolicy::RP)),                                            \
                           framework::dataset::make("FixedPointPosition", FPP_START, FPP_END)))                                        \
    {                                                                                                                                  \
        validate(Accessor(_target), _reference);                                                                                       \
    }

#define FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(TEST_NAME, FIXTURE, MODE, SHAPES, DT1, DT2, RP, FPP, TOLERANCE)  \
    FIXTURE_DATA_TEST_CASE(TEST_NAME, NEFixedPointPixelWiseMultiplication##FIXTURE, framework::DatasetMode::MODE,                  \
                           combine(combine(combine(combine(combine(combine(                                                        \
                           datasets::SHAPES,                                                                                       \
                           framework::dataset::make("DataType1", DataType::DT1)),                                                  \
                           framework::dataset::make("DataType2", DataType::DT2)),                                                  \
                           framework::dataset::make("Scale", 1.f / static_cast<float>(1 << (FPP)))),                               \
                           datasets::ConvertPolicies()),                                                                           \
                           framework::dataset::make("RoundingPolicy", RoundingPolicy::RP)),                                        \
                           framework::dataset::make("FixedPointPosition", FPP)))                                                   \
    {                                                                                                                              \
        validate(Accessor(_target), _reference, AbsoluteTolerance<float>(TOLERANCE), 0.f);                                         \
    }
// clang-format on
// *INDENT-ON*
} // namespace

template <typename T>
using NEFixedPointPixelWiseMultiplicationFixture = FixedPointPixelWiseMultiplicationValidationFixture<Tensor, Accessor, NEPixelWiseMultiplication, T>;

TEST_SUITE(NEON)
TEST_SUITE(FixedPointPixelWiseMultiplication)

TEST_SUITE(QS8)

TEST_SUITE(Scale255)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, scale_255, TO_NEAREST_UP, 1, 7)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, scale_255, TO_NEAREST_UP, 1, 7)
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, scale_unity, TO_ZERO, 1, 7)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, scale_unity, TO_ZERO, 1, 7)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther1, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, TO_ZERO, 1, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther2, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, TO_ZERO, 2, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther3, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, TO_ZERO, 3, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther4, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, TO_ZERO, 4, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther5, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, TO_ZERO, 5, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther6, Fixture<qint8_t>, PRECOMMIT, SmallShapes(), QS8, QS8, TO_ZERO, 6, tolerance)

FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunLargeOther1, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, TO_ZERO, 1, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunLargeOther2, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, TO_ZERO, 2, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunLargeOther3, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, TO_ZERO, 3, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunLargeOther4, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, TO_ZERO, 4, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunLargeOther5, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, TO_ZERO, 5, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunLargeOther6, Fixture<qint8_t>, NIGHTLY, LargeShapes(), QS8, QS8, TO_ZERO, 6, tolerance)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // QS8

TEST_SUITE(QS16)

TEST_SUITE(Scale255)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, scale_255, TO_NEAREST_UP, 1, 15)
TEST_SUITE_END() // Scale255

TEST_SUITE(ScaleUnity)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunSmall, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, scale_unity, TO_ZERO, 1, 15)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE(RunLarge, Fixture<qint16_t>, NIGHTLY, LargeShapes(), QS16, QS16, scale_unity, TO_ZERO, 1, 15)
TEST_SUITE_END() // ScaleUnity

TEST_SUITE(ScaleOther)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther1, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 1, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther2, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 2, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther3, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 3, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther4, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 4, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther5, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 5, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther6, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 6, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther7, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 7, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther8, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 8, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther9, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 9, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther10, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 10, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther11, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 11, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther12, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 12, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther13, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 13, tolerance)
FP_PIXEL_WISE_MULTIPLICATION_FIXTURE_DATA_TEST_CASE_OTHER(RunSmallOther14, Fixture<qint16_t>, PRECOMMIT, SmallShapes(), QS16, QS16, TO_ZERO, 14, tolerance)
TEST_SUITE_END() // ScaleOther

TEST_SUITE_END() // QS16

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
