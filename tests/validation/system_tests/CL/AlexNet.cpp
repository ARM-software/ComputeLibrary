#ifdef INTERNAL_ONLY //FIXME Delete this file before the release
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
#include "CL/CLAccessor.h"
#include "validation/Validation.h"

#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"

#include "model_objects/AlexNet.h"

#include <array>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::cl;
using namespace arm_compute::test::validation;

namespace
{
using CLAlexNetModel = model_objects::AlexNet<ICLTensor,
      CLTensor,
      CLSubTensor,
      CLAccessor,
      CLActivationLayer,
      CLConvolutionLayer,
      CLFullyConnectedLayer,
      CLNormalizationLayer,
      CLPoolingLayer,
      CLSoftmaxLayer>;
std::vector<unsigned int> compute_alexnet(unsigned int batches, std::string input_file)
{
    std::vector<std::string> weight_files = { "cnn_data/alexnet_model/conv1_w.dat",
                                              "cnn_data/alexnet_model/conv2_w.dat",
                                              "cnn_data/alexnet_model/conv3_w.dat",
                                              "cnn_data/alexnet_model/conv4_w.dat",
                                              "cnn_data/alexnet_model/conv5_w.dat",
                                              "cnn_data/alexnet_model/fc6_w.dat",
                                              "cnn_data/alexnet_model/fc7_w.dat",
                                              "cnn_data/alexnet_model/fc8_w.dat"
                                            };

    std::vector<std::string> bias_files = { "cnn_data/alexnet_model/conv1_b.dat",
                                            "cnn_data/alexnet_model/conv2_b.dat",
                                            "cnn_data/alexnet_model/conv3_b.dat",
                                            "cnn_data/alexnet_model/conv4_b.dat",
                                            "cnn_data/alexnet_model/conv5_b.dat",
                                            "cnn_data/alexnet_model/fc6_b.dat",
                                            "cnn_data/alexnet_model/fc7_b.dat",
                                            "cnn_data/alexnet_model/fc8_b.dat"
                                          };
    CLAlexNetModel network{};
    network.init_weights(batches);
    network.build();
    network.allocate();
    network.fill(weight_files, bias_files);
    network.feed(std::move(input_file));
    network.run();

    return network.get_classifications();
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(SYSTEM_TESTS)
BOOST_AUTO_TEST_SUITE(CL)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_AUTO_TEST_CASE(AlexNet)
{
    // Compute alexnet
    std::vector<unsigned int> classified_labels = compute_alexnet(1, "cnn_data/imagenet_data/shark.dat");

    // Expected labels
    std::vector<unsigned int> expected_labels = { 2 };

    // Validate labels
    validate(classified_labels, expected_labels);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
#endif /* INTERNAL_ONLY */
