/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/PoolingLayerFixture.h"
#include "tests/datasets/system_tests/alexnet/AlexNetPoolingLayerDataset.h"
#include "tests/datasets/system_tests/googlenet/inceptionv1/GoogLeNetInceptionV1PoolingLayerDataset.h"
#include "tests/datasets/system_tests/googlenet/inceptionv4/GoogLeNetInceptionV4PoolingLayerDataset.h"
#include "tests/datasets/system_tests/lenet5/LeNet5PoolingLayerDataset.h"
#include "tests/datasets/system_tests/squeezenet/SqueezeNetPoolingLayerDataset.h"
#include "tests/datasets/system_tests/vgg/vgg16/VGG16PoolingLayerDataset.h"
#include "tests/datasets/system_tests/yolo/v2/YOLOV2PoolingLayerDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const auto data_types = framework::dataset::make("DataType", { DataType::F16, DataType::F32, DataType::QASYMM8 });
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const auto data_types = framework::dataset::make("DataType", { DataType::F32, DataType::QASYMM8 });
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace

using NEPoolingLayerFixture = PoolingLayerFixture<Tensor, NEPoolingLayer, Accessor>;

TEST_SUITE(NEON)

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetPoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::AlexNetPoolingLayerDataset(), data_types), framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(LeNet5PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::LeNet5PoolingLayerDataset(), data_types), framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1PoolingLayerDataset(), data_types), framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4PoolingLayerDataset(), data_types), framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetPoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetPoolingLayerDataset(), data_types), framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(VGG16PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::VGG16PoolingLayerDataset(), data_types), framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(YOLOV2PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::YOLOV2PoolingLayerDataset(), data_types), framework::dataset::make("Batches", 1)));

TEST_SUITE(NIGHTLY)
REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetPoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::AlexNetPoolingLayerDataset(), data_types), framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(LeNet5PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::LeNet5PoolingLayerDataset(), data_types), framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1PoolingLayerDataset(), data_types), framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4PoolingLayerDataset(), data_types), framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetPoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetPoolingLayerDataset(), data_types), framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(VGG16PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::VGG16PoolingLayerDataset(), data_types), framework::dataset::make("Batches", { 2 })));

REGISTER_FIXTURE_DATA_TEST_CASE(YOLOV2PoolingLayer, NEPoolingLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::YOLOV2PoolingLayerDataset(), data_types), framework::dataset::make("Batches", { 4, 8 })));
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace test
} // namespace arm_compute
