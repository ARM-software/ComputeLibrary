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
#include "NEON/NEAccessor.h"
#include "validation/Validation.h"

#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"

#include "model_objects/LeNet5.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
using NELeNet5Model = model_objects::LeNet5<Tensor,
      NEAccessor,
      NEActivationLayer,
      NEConvolutionLayer,
      NEFullyConnectedLayer,
      NEPoolingLayer,
      NESoftmaxLayer>;
std::vector<unsigned int> compute_lenet5(unsigned int batches, std::string input_file)
{
    std::vector<std::string> weight_files = { "cnn_data/lenet_model/conv1_w.dat",
                                              "cnn_data/lenet_model/conv2_w.dat",
                                              "cnn_data/lenet_model/ip1_w.dat",
                                              "cnn_data/lenet_model/ip2_w.dat"
                                            };

    std::vector<std::string> bias_files = { "cnn_data/lenet_model/conv1_b.dat",
                                            "cnn_data/lenet_model/conv2_b.dat",
                                            "cnn_data/lenet_model/ip1_b.dat",
                                            "cnn_data/lenet_model/ip2_b.dat"
                                          };
    NELeNet5Model network{};
    network.build(batches);
    network.fill(weight_files, bias_files);
    network.feed(std::move(input_file));
    network.run();

    return network.get_classifications();
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(SYSTEM_TESTS)
BOOST_AUTO_TEST_SUITE(NEON)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_AUTO_TEST_CASE(LeNet5)
{
    // Compute alexnet
    std::vector<unsigned int> classified_labels = compute_lenet5(10, "cnn_data/mnist_data/input100.dat");

    // Expected labels
    std::vector<unsigned int> expected_labels = { 7, 2, 1, 0, 4, 1, 4, 9, 5, 9 };

    // Validate labels
    validate(classified_labels, expected_labels);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
#endif /* INTERNAL_ONLY */
