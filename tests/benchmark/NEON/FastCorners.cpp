/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/NEON/functions/NEFastCorners.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/FastCornersFixture.h"
#include "tests/datasets/ImageFileDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
namespace
{
const auto threshold       = framework::dataset::make("Threshold", { 64.f });                   // valid range (0.0 ≤ threshold < 256.0)
const auto border_mode     = framework::dataset::make("BorderMode", { BorderMode::UNDEFINED }); // NOTE: only BorderMode::UNDEFINED is implemented
const auto suppress_nonmax = framework::dataset::make("SuppressNonMax", { false, true });
} // namespace

using NEFastCornersFixture = FastCornersFixture<Tensor, NEFastCorners, Accessor, KeyPointArray>;

TEST_SUITE(NEON)
TEST_SUITE(FastCorners)

// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, NEFastCornersFixture, framework::DatasetMode::PRECOMMIT,
                                combine(combine(combine(combine(
                                datasets::SmallImageFiles(),
                                framework::dataset::make("Format", { Format::U8 })),
                                threshold),
                                suppress_nonmax),
                                border_mode));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, NEFastCornersFixture, framework::DatasetMode::NIGHTLY,
                                combine(combine(combine(combine(
                                datasets::LargeImageFiles(),
                                framework::dataset::make("Format", { Format::U8 })),
                                threshold),
                                suppress_nonmax),
                                border_mode));
// clang-format on
// *INDENT-ON*

TEST_SUITE_END() // FastCorners
TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
