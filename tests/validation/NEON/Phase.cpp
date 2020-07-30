/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPhase.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PhaseFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<uint8_t> tolerance_value(1);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Phase)

template <typename T>
using NEPhaseFixture = PhaseValidationFixture<Tensor, Accessor, NEPhase, T>;

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPhaseFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::Small2DShapes(), framework::dataset::make("Format", Format::S16)),
                                                                                                     framework::dataset::make("PhaseType", { PhaseType::UNSIGNED, PhaseType::SIGNED })))
{
    // Validate output
    validate_wrap(Accessor(_target), _reference, tolerance_value, 0);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPhaseFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::Large2DShapes(), framework::dataset::make("Format", Format::S16)),
                                                                                                   framework::dataset::make("PhaseType", { PhaseType::UNSIGNED, PhaseType::SIGNED })))
{
    // Validate output
    validate_wrap(Accessor(_target), _reference, tolerance_value, 0);
}
TEST_SUITE_END() // S16

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
