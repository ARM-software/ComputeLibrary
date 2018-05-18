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
#include "arm_compute/runtime/NEON/functions/NEGaussianPyramid.h"
#include "arm_compute/runtime/NEON/functions/NEOpticalFlow.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/ArrayAccessor.h"
#include "tests/benchmark/fixtures/OpticalFlowFixture.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ImageFileDatasets.h"
#include "tests/datasets/OpticalFlowDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
using NEOpticalFlowFixture = OpticalFlowFixture<Tensor, NEOpticalFlow, Accessor, KeyPointArray, ArrayAccessor<KeyPoint>, Pyramid, NEGaussianPyramidHalf>;

TEST_SUITE(NEON)
TEST_SUITE(OpticalFlow)

// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, NEOpticalFlowFixture, framework::DatasetMode::PRECOMMIT,
                                combine(combine(
                                datasets::SmallOpticalFlowDataset(),
                                framework::dataset::make("Format", Format::U8)),
                                datasets::BorderModes()));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, NEOpticalFlowFixture, framework::DatasetMode::NIGHTLY,
                                combine(combine(
                                datasets::LargeOpticalFlowDataset(),
                                framework::dataset::make("Format", Format::U8)),
                                datasets::BorderModes()));
// clang-format on
// *INDENT-ON*

TEST_SUITE_END() // OpticalFlow
TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
