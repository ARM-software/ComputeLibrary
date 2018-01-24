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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLScale.h"
#include "tests/CL/CLAccessor.h"
#include "tests/benchmark/fixtures/ScaleFixture.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/SamplingPolicyDataset.h"
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
const auto interpolation_types = framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR });
} // namespace

using CLScaleFixture = ScaleFixture<CLTensor, CLScale, CLAccessor>;

TEST_SUITE(CL)
TEST_SUITE(Scale)
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallImageShapes(), framework::dataset::make("DataType", { DataType::F16, DataType::F32 })),
                                                                                                                     interpolation_types),
                                                                                                             datasets::BorderModes()),
                                                                                                     datasets::SamplingPolicies()));
REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeImageShapes(), framework::dataset::make("DataType", { DataType::F16, DataType::F32 })),
                                                                                                                   interpolation_types),
                                                                                                           datasets::BorderModes()),
                                                                                                   datasets::SamplingPolicies()));
TEST_SUITE_END() // Scale
TEST_SUITE_END() // CL
} // namespace benchmark
} // namespace test
} // namespace arm_compute
