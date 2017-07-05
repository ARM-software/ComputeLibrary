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
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/NEON/NEAccessor.h"
#include "tests/TypePrinter.h"
#include "tests/datasets_new/AlexNetFullyConnectedLayerDataset.h"
#include "tests/datasets_new/GoogLeNetFullyConnectedLayerDataset.h"
#include "tests/datasets_new/LeNet5FullyConnectedLayerDataset.h"
#include "tests/fixtures_new/FullyConnectedLayerFixture.h"

namespace arm_compute
{
namespace test
{
using NEFullyConnectedLayerFixture = FullyConnectedLayerFixture<Tensor, NEFullyConnectedLayer, neon::NEAccessor>;

TEST_SUITE(NEON)

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetFullyConnectedLayer, NEFullyConnectedLayerFixture,
                                framework::dataset::combine(framework::dataset::combine(datasets::AlexNetFullyConnectedLayerDataset(),
                                                                                        framework::dataset::make("Data type", { DataType::F32, DataType::QS8 })),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(LeNet5FullyConnectedLayer, NEFullyConnectedLayerFixture,
                                framework::dataset::combine(framework::dataset::combine(datasets::LeNet5FullyConnectedLayerDataset(),
                                                                                        framework::dataset::make("Data type", DataType::F32)),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetFullyConnectedLayer, NEFullyConnectedLayerFixture,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetFullyConnectedLayerDataset(),
                                                                                        framework::dataset::make("Data type", DataType::F32)),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

TEST_SUITE_END()
} // namespace test
} // namespace arm_compute
