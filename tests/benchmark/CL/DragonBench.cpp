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
#ifdef INTERNAL_ONLY

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"

#include "tests/benchmark/DragonBenchConfigs.h"

#include "tests/CL/CLAccessor.h"
#include "tests/benchmark/fixtures/DragonBenchFixture.h"
#include "tests/datasets/DragonBenchDataset.h"
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
// Common parameters
const auto data_types = framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F16, DataType::F32 });

// Common DragonBench parameters
auto CommonConv2DParamsNoBias = combine(combine(data_types, framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                        framework::dataset::make("HasBias", { false }));
auto CommonConv2DParamsBias = combine(combine(data_types, framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                      framework::dataset::make("HasBias", { true }));

// Common DragonBench FC parameters
auto CommonFCParams = combine(data_types, framework::dataset::make("HasBias", { false }));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DragonBench)

TEST_SUITE(Conv2D)
using CLDragonBenchConv2DFixture = DragonBenchConv2DFixture<CLTensor, CLConvolutionLayer, CLAccessor, Conv2D>;

REGISTER_FIXTURE_DATA_TEST_CASE(SilverWing,
                                CLDragonBenchConv2DFixture,
                                framework::DatasetMode::ALL,
                                combine(datasets::DragonBenchDataset<Conv2D_Configs, Conv2D>(silverwing_cfgs), CommonConv2DParamsNoBias));

REGISTER_FIXTURE_DATA_TEST_CASE(SunFyre,
                                CLDragonBenchConv2DFixture,
                                framework::DatasetMode::ALL,
                                combine(datasets::DragonBenchDataset<Conv2D_Configs, Conv2D>(sunfyre_cfgs), CommonConv2DParamsNoBias));

REGISTER_FIXTURE_DATA_TEST_CASE(Syrax,
                                CLDragonBenchConv2DFixture,
                                framework::DatasetMode::ALL,
                                combine(datasets::DragonBenchDataset<Conv2D_Configs, Conv2D>(syrax_cfgs), CommonConv2DParamsNoBias));

TEST_SUITE(Nightly)
REGISTER_FIXTURE_DATA_TEST_CASE(SilverWing,
                                CLDragonBenchConv2DFixture,
                                framework::DatasetMode::NIGHTLY,
                                combine(datasets::DragonBenchDataset<Conv2D_Configs, Conv2D>(silverwing_cfgs), CommonConv2DParamsBias));

REGISTER_FIXTURE_DATA_TEST_CASE(SunFyre,
                                CLDragonBenchConv2DFixture,
                                framework::DatasetMode::NIGHTLY,
                                combine(datasets::DragonBenchDataset<Conv2D_Configs, Conv2D>(sunfyre_cfgs), CommonConv2DParamsBias));

REGISTER_FIXTURE_DATA_TEST_CASE(Syrax,
                                CLDragonBenchConv2DFixture,
                                framework::DatasetMode::NIGHTLY,
                                combine(datasets::DragonBenchDataset<Conv2D_Configs, Conv2D>(syrax_cfgs), CommonConv2DParamsBias));
TEST_SUITE_END() // Nightly
TEST_SUITE_END() // Conv2D

TEST_SUITE(FullyConnected)
using CLDragonBenchFCFixture = DragonBenchFCFixture<CLTensor, CLFullyConnectedLayer, CLAccessor, Fully_Connected>;

REGISTER_FIXTURE_DATA_TEST_CASE(DreamFyre,
                                CLDragonBenchFCFixture,
                                framework::DatasetMode::ALL,
                                combine(datasets::DragonBenchDataset<Fully_Connected_Configs, Fully_Connected>(dreamfyre_cfgs), CommonFCParams));
TEST_SUITE_END() // FullyConnected

TEST_SUITE_END() // DragonBench
TEST_SUITE_END() // CL
} // namespace benchmark
} // namespace test
} // namespace arm_compute

#endif // INTERNAL_ONLY
