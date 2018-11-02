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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/benchmark/fixtures/ConvolutionLayerFixture.h"
#include "tests/benchmark/fixtures/WinogradConvolutionLayerFixture.h"
#include "tests/datasets/system_tests/alexnet/AlexNetConvolutionLayerDataset.h"
#include "tests/datasets/system_tests/googlenet/inceptionv1/GoogLeNetInceptionV1ConvolutionLayerDataset.h"
#include "tests/datasets/system_tests/googlenet/inceptionv4/GoogLeNetInceptionV4ConvolutionLayerDataset.h"
#include "tests/datasets/system_tests/lenet5/LeNet5ConvolutionLayerDataset.h"
#include "tests/datasets/system_tests/mobilenet/MobileNetConvolutionLayerDataset.h"
#include "tests/datasets/system_tests/squeezenet/SqueezeNetConvolutionLayerDataset.h"
#include "tests/datasets/system_tests/vgg/vgg16/VGG16ConvolutionLayerDataset.h"
#include "tests/datasets/system_tests/yolo/v2/YOLOV2ConvolutionLayerDataset.h"
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
const auto data_types = framework::dataset::make("DataType", { DataType::F16, DataType::F32, DataType::QASYMM8 });
} // namespace

using CLGEMMConvolutionLayerFixture = ConvolutionLayerFixture<CLTensor, CLGEMMConvolutionLayer, CLAccessor>;

TEST_SUITE(CL)

using CLWinogradLayerFixture = WinogradConvolutionLayerFixture<CLTensor, CLWinogradConvolutionLayer, CLAccessor>;

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetWinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::AlexNetWinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1WinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1WinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4WinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4WinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetWinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetWinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::AlexNetConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(LeNet5ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::LeNet5ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo())),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

REGISTER_FIXTURE_DATA_TEST_CASE(MobileNetConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::MobileNetConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", 1)));

TEST_SUITE(NIGHTLY)
REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::AlexNetConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(LeNet5ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::LeNet5ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo())),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 4, 8 })));

// 8 batches use about 1.8GB of memory which is too much for most devices!
REGISTER_FIXTURE_DATA_TEST_CASE(VGG16ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::VGG16ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo",
                                                                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 1, 4 })));

REGISTER_FIXTURE_DATA_TEST_CASE(YOLOV2ConvolutionLayer, CLGEMMConvolutionLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::YOLOV2ConvolutionLayerDataset(), framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                        data_types),
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNetWinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::AlexNetWinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1WinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1WinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV4WinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV4WinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", { 4, 8 })));

REGISTER_FIXTURE_DATA_TEST_CASE(SqueezeNetWinogradLayer, CLWinogradLayerFixture, framework::DatasetMode::NIGHTLY,
                                framework::dataset::combine(framework::dataset::combine(framework::dataset::combine(datasets::SqueezeNetWinogradLayerDataset(),
                                                                                                                    framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                            framework::dataset::make("Batches", { 4, 8 })));
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace benchmark
} // namespace test
} // namespace arm_compute
