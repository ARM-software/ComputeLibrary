/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/benchmark/fixtures/ElementWiseUnaryFixture.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
TEST_SUITE(NEON)
TEST_SUITE(SinLayer)

template <typename T>
using NESinLayerFixture = SinBenchmarkFixture<Tensor, Accessor, NESinLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, NESinLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                              DataType::F16)));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, NESinLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                            DataType::F16)));

TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE(FP32)
REGISTER_FIXTURE_DATA_TEST_CASE(RunSmall, NESinLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                         DataType::F32)));

REGISTER_FIXTURE_DATA_TEST_CASE(RunLarge, NESinLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                             DataType::F32)));
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // SinLayer
TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
