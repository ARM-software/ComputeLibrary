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
#include "arm_compute/runtime/NEON/functions/NECannyEdge.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/ArrayAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ImageFileDatasets.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CannyEdgeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/* Allowed ratio of mismatches between target and reference (1.0 = 100%) */
const float allowed_mismatch_ratio = 0.1f;

const auto data = combine(framework::dataset::make("GradientSize", { 3, 5, 7 }),
                          combine(framework::dataset::make("Normalization", { MagnitudeType::L1NORM, MagnitudeType::L2NORM }), datasets::BorderModes()));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(CannyEdge)

template <typename T>
using NECannyEdgeFixture = CannyEdgeValidationFixture<Tensor, Accessor, KeyPointArray, NECannyEdge, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NECannyEdgeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SmallImageFiles(), data), framework::dataset::make("Format", Format::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, AbsoluteTolerance<uint8_t>(0), allowed_mismatch_ratio);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NECannyEdgeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeImageFiles(), data), framework::dataset::make("Format", Format::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, AbsoluteTolerance<uint8_t>(0), allowed_mismatch_ratio);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
