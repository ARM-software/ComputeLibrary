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
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/SubTensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/networks/AlexNetNetwork.h"
#include "tests/validation/Validation.h"

#include <string>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
using NEAlexNetModel = networks::AlexNetNetwork<ITensor,
      Tensor,
      SubTensor,
      Accessor,
      NEActivationLayer,
      NEConvolutionLayer,
      NEDirectConvolutionLayer,
      NEFullyConnectedLayer,
      NENormalizationLayer,
      NEPoolingLayer,
      NESoftmaxLayer>;
std::vector<unsigned int> compute_alexnet(DataType dt, unsigned int batches, std::string input_file)
{
    std::vector<std::string> weight_files = { "cnn_data/alexnet_model/conv1_w.npy",
                                              "cnn_data/alexnet_model/conv2_w.npy",
                                              "cnn_data/alexnet_model/conv3_w.npy",
                                              "cnn_data/alexnet_model/conv4_w.npy",
                                              "cnn_data/alexnet_model/conv5_w.npy",
                                              "cnn_data/alexnet_model/fc6_w.npy",
                                              "cnn_data/alexnet_model/fc7_w.npy",
                                              "cnn_data/alexnet_model/fc8_w.npy"
                                            };

    std::vector<std::string> bias_files = { "cnn_data/alexnet_model/conv1_b.npy",
                                            "cnn_data/alexnet_model/conv2_b.npy",
                                            "cnn_data/alexnet_model/conv3_b.npy",
                                            "cnn_data/alexnet_model/conv4_b.npy",
                                            "cnn_data/alexnet_model/conv5_b.npy",
                                            "cnn_data/alexnet_model/fc6_b.npy",
                                            "cnn_data/alexnet_model/fc7_b.npy",
                                            "cnn_data/alexnet_model/fc8_b.npy"
                                          };
    NEAlexNetModel network{};

    network.init(dt, 4, batches);
    network.build();
    network.allocate();
    network.fill(weight_files, bias_files);
    network.feed(std::move(input_file));
    network.run();

    return network.get_classifications();
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(SYSTEM_TESTS)

TEST_CASE(AlexNet, framework::DatasetMode::PRECOMMIT)
{
    // Compute alexnet
    std::vector<unsigned int> classified_labels = compute_alexnet(DataType::F32, 1, "cnn_data/imagenet_data/cat.npy");

    // Expected labels
    std::vector<unsigned int> expected_labels = { 281 };

    // Validate labels
    validate(classified_labels, expected_labels);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
