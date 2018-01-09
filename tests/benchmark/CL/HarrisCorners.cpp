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
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLHarrisCorners.h"
#include "tests/CL/CLAccessor.h"
#include "tests/benchmark/fixtures/HarrisCornersFixture.h"
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
const auto threshold     = framework::dataset::make("Threshold", { 0.00115f });
const auto min_dist      = framework::dataset::make("MinDist", { 2.f });
const auto sensitivity   = framework::dataset::make("Sensitivity", { 0.04f });
const auto gradient_size = framework::dataset::make("GradientSize", { 3, 5, 7 });
const auto block_size    = framework::dataset::make("BlockSize", { 3, 5, 7 });
const auto border_mode   = framework::dataset::make("BorderMode", { BorderMode::UNDEFINED, BorderMode::CONSTANT, BorderMode::REPLICATE });
} // namespace

using CLHarrisCornersFixture = HarrisCornersFixture<CLTensor, CLHarrisCorners, CLAccessor, CLKeyPointArray>;

TEST_SUITE(CL)
TEST_SUITE(HarrisCorners)

REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLHarrisCornersFixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(combine(combine(datasets::SmallImageFiles(),
                                                                                                                     framework::dataset::make("Format", { Format::U8 })),
                                                                                                                     threshold),
                                                                                                                     min_dist),
                                                                                                                     sensitivity),
                                                                                                                     gradient_size),
                                                                                                                     block_size),
                                                                                                                     border_mode),
                                                                                                             framework::dataset::make("UseFP16", { false })));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLHarrisCornersFixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(combine(datasets::LargeImageFiles(),
                                                                                                                   framework::dataset::make("Format", { Format::U8 })),
                                                                                                                   threshold),
                                                                                                                   min_dist),
                                                                                                                   sensitivity),
                                                                                                                   gradient_size),
                                                                                                                   block_size),
                                                                                                                   border_mode),
                                                                                                           framework::dataset::make("UseFP16", { false })));

TEST_SUITE_END() // HarrisCorners
TEST_SUITE_END() // CL
} // namespace benchmark
} // namespace test
} // namespace arm_compute