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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEWarpPerspective.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WarpPerspectiveFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<uint8_t> tolerance_value(1);
constexpr float                      tolerance_number = 0.2f;
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(WarpPerspective)

template <typename T>
using NEWarpPerspectiveFixture = WarpPerspectiveValidationFixture<Tensor, Accessor, NEWarpPerspective, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEWarpPerspectiveFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                     DataType::U8)),
                                                                                                                     framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                             datasets::BorderModes()))
{
    validate(Accessor(_target), _reference, _valid_mask, tolerance_value, tolerance_number);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWarpPerspectiveFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                     DataType::U8)),
                                                                                                                     framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                             datasets::BorderModes()))
{
    validate(Accessor(_target), _reference, _valid_mask, tolerance_value, tolerance_number);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
