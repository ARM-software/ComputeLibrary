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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEMagnitude.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/MagnitudeFixture.h"
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
const auto magnitude_types = framework::dataset::make("MagnitudeType", { MagnitudeType::L1NORM, MagnitudeType::L2NORM });
} // namespace

using NEMagnitudeFixture = MagnitudeFixture<Tensor, NEMagnitude, Accessor>;

TEST_SUITE(NEON)
TEST_SUITE(Magnitude)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, NEMagnitudeFixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallImageShapes(), framework::dataset::make("Format", { Format::S16 })),
                                                                                                                 magnitude_types),
                                                                                                         framework::dataset::make("UseFP16", { true })));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, NEMagnitudeFixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeImageShapes(), framework::dataset::make("Format", { Format::S16 })),
                                                                                                               magnitude_types),
                                                                                                       framework::dataset::make("UseFP16", { true })));
TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE(S16)
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, NEMagnitudeFixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallImageShapes(), framework::dataset::make("Format", { Format::S16 })),
                                                                                                                 magnitude_types),
                                                                                                         framework::dataset::make("UseFP16", { false })));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, NEMagnitudeFixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeImageShapes(), framework::dataset::make("Format", { Format::S16 })),
                                                                                                               magnitude_types),
                                                                                                       framework::dataset::make("UseFP16", { false })));
TEST_SUITE_END() // S16
TEST_SUITE_END() // Magnitude
TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
