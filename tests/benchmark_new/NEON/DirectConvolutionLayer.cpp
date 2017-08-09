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
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/NEON/Accessor.h"
#include "tests/TypePrinter.h"
#include "tests/datasets_new/AlexNetConvolutionLayerDataset.h"
#include "tests/datasets_new/DirectConvolutionLayerDataset.h"
#include "tests/datasets_new/GoogLeNetConvolutionLayerDataset.h"
#include "tests/datasets_new/SqueezeNetConvolutionLayerDataset.h"
#include "tests/fixtures_new/ConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace
{
#ifdef ARM_COMPUTE_ENABLE_F16
const auto data_types = framework::dataset::make("DataType", { DataType::QS8, DataType::F16, DataType::F32 });
#else  /* ARM_COMPUTE_ENABLE_F16 */
const auto data_types = framework::dataset::make("DataType", { DataType::QS8, DataType::F32 });
#endif /* ARM_COMPUTE_ENABLE_F16 */
} // namespace

using NEConvolutionLayerFixture = ConvolutionLayerFixture<Tensor, NEDirectConvolutionLayer, Accessor>;

TEST_SUITE(NEON)

REGISTER_FIXTURE_DATA_TEST_CASE(DirectConvolutionLayer, NEConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::DirectConvolutionLayerDataset(), data_types),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetDirectConvolutionLayer, NEConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::AlexNetDirectConvolutionLayerDataset(), data_types),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetDirectConvolutionLayer, NEConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetDirectConvolutionLayerDataset(), data_types),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetDirectConvolutionLayer, NEConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetConvolutionLayerDataset(), data_types),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

TEST_SUITE_END()
} // namespace test
} // namespace arm_compute
