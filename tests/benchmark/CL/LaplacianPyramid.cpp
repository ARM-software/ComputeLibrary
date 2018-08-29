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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLLaplacianPyramid.h"
#include "tests/CL/CLAccessor.h"
#include "tests/benchmark/fixtures/LaplacianPyramidFixture.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
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
const auto small_laplacian_pyramid_levels = framework::dataset::make("NumLevels", 2, 3);
const auto large_laplacian_pyramid_levels = framework::dataset::make("NumLevels", 2, 5);

const auto formats = combine(framework::dataset::make("FormatIn", Format::U8), framework::dataset::make("FormatOut", Format::S16));
} // namespace

using CLLaplacianPyramidFixture = LaplacianPyramidFixture<CLTensor, CLLaplacianPyramid, CLAccessor, CLPyramid>;

TEST_SUITE(CL)
TEST_SUITE(LaplacianPyramid)

// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLLaplacianPyramidFixture, framework::DatasetMode::PRECOMMIT,
                                combine(combine(combine(
                                datasets::Medium2DShapes(),
                                datasets::BorderModes()),
                                small_laplacian_pyramid_levels),
                                formats));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLLaplacianPyramidFixture, framework::DatasetMode::NIGHTLY,
                                combine(combine(combine(
                                datasets::Large2DShapes(),
                                datasets::BorderModes()),
                                large_laplacian_pyramid_levels),
                                formats));
// clang-format on
// *INDENT-ON*

TEST_SUITE_END() // LaplacianPyramid
TEST_SUITE_END() // CL
} // namespace benchmark
} // namespace test
} // namespace arm_compute
