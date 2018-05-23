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
#include "arm_compute/runtime/CL/CLHOG.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLHOGDescriptor.h"
#include "tests/CL/CLAccessor.h"
#include "tests/benchmark/fixtures/HOGDescriptorFixture.h"
#include "tests/datasets/HOGDescriptorDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
using CLHOGDescriptorFixture = HOGDescriptorFixture<CLTensor, CLHOG, CLHOGDescriptor, CLAccessor>;

TEST_SUITE(CL)
TEST_SUITE(HOGDescriptor)

// *INDENT-OFF*
// clang-format off
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLHOGDescriptorFixture, framework::DatasetMode::PRECOMMIT,
                                combine(combine(
                                datasets::SmallHOGDescriptorDataset(),
                                framework::dataset::make("Format", { Format::U8 })),
                                framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLHOGDescriptorFixture, framework::DatasetMode::NIGHTLY,
                                combine(combine(
                                datasets::LargeHOGDescriptorDataset(),
                                framework::dataset::make("Format", { Format::U8 })),
                                framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})));
// clang-format on
// *INDENT-ON*

TEST_SUITE_END() // HOGDescriptor
TEST_SUITE_END() // CL
} // namespace benchmark
} // namespace test
} // namespace arm_compute
