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
#include "arm_compute/runtime/MultiHOG.h"
#include "arm_compute/runtime/NEON/functions/NEHOGMultiDetection.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/ArrayAccessor.h"
#include "tests/NEON/HOGAccessor.h"
#include "tests/benchmark/fixtures/HOGMultiDetectionFixture.h"
#include "tests/datasets/HOGMultiDetectionDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
// *INDENT-OFF*
// clang-format off
using NEHOGMultiDetectionFixture = HOGMultiDetectionFixture<Tensor,
                                                            HOG,
                                                            MultiHOG,
                                                            DetectionWindowArray,
                                                            Size2DArray,
                                                            NEHOGMultiDetection,
                                                            Accessor,
                                                            HOGAccessor,
                                                            ArrayAccessor<Size2D>>;
TEST_SUITE(NEON)
TEST_SUITE(HOGMultiDetection)

REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, NEHOGMultiDetectionFixture, framework::DatasetMode::PRECOMMIT,
                                combine(combine(combine(
                                datasets::SmallHOGMultiDetectionDataset(),
                                framework::dataset::make("Format", Format::U8)),
                                framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})),
                                framework::dataset::make("NonMaximaSuppression", {false, true})));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, NEHOGMultiDetectionFixture, framework::DatasetMode::NIGHTLY,
                                combine(combine(combine(
                                datasets::LargeHOGMultiDetectionDataset(),
                                framework::dataset::make("Format", Format::U8)),
                                framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})),
                                framework::dataset::make("NonMaximaSuppression", {false, true})));
// clang-format on
// *INDENT-ON*

TEST_SUITE_END() // HOGMultiDetection
TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
