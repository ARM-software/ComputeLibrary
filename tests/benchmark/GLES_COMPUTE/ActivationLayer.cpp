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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCActivationLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/benchmark/fixtures/ActivationLayerFixture.h"
#include "tests/datasets/system_tests/alexnet/AlexNetActivationLayerDataset.h"
#include "tests/datasets/system_tests/googlenet/inceptionv1/GoogLeNetInceptionV1ActivationLayerDataset.h"
#include "tests/datasets/system_tests/googlenet/inceptionv4/GoogLeNetInceptionV4ActivationLayerDataset.h"
#include "tests/datasets/system_tests/lenet5/LeNet5ActivationLayerDataset.h"
#include "tests/datasets/system_tests/mobilenet/MobileNetActivationLayerDataset.h"
#include "tests/datasets/system_tests/squeezenet/SqueezeNetActivationLayerDataset.h"
#include "tests/datasets/system_tests/vgg/vgg16/VGG16ActivationLayerDataset.h"
#include "tests/datasets/system_tests/yolo/v2/YOLOV2ActivationLayerDataset.h"
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
const auto data_types           = framework::dataset::make("DataType", { DataType::F16, DataType::F32 });
const auto data_types_mobilenet = framework::dataset::make("DataType", { DataType::F16, DataType::F32 });
} // namespace

using GCActivationLayerFixture = ActivationLayerFixture<GCTensor, GCActivationLayer, GCAccessor>;

TEST_SUITE(GC)

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::AlexNetActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(LeNet5ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::LeNet5ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(MobileNetActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::MobileNetActivationLayerDataset(),
                                                                                        data_types_mobilenet),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(VGG16ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::VGG16ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(YOLOV2ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::YOLOV2ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

TEST_SUITE(NIGHTLY)
REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::AlexNetActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(LeNet5ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::LeNet5ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(MobileNetActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::MobileNetActivationLayerDataset(),
                                                                                        data_types_mobilenet),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(VGG16ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::VGG16ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(YOLOV2ActivationLayer, GCActivationLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(datasets::YOLOV2ActivationLayerDataset(),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace benchmark
} // namespace test
} // namespace arm_compute
