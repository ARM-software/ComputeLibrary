/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute::utils;
using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement VGG19's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
class GraphVGG19Example : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        constexpr float mean_r = 123.68f;  /* Mean value to subtract from red channel */
        constexpr float mean_g = 116.779f; /* Mean value to subtract from green channel */
        constexpr float mean_b = 103.939f; /* Mean value to subtract from blue channel */

        // Set target. 0 (NEON), 1 (OpenCL). By default it is NEON
        TargetHint            target_hint      = set_target_hint(argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0);
        ConvolutionMethodHint convolution_hint = ConvolutionMethodHint::DIRECT;

        // Parse arguments
        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [target] [path_to_data] [image] [labels]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 2)
        {
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " [path_to_data] [image] [labels]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 3)
        {
            data_path = argv[2];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " [image] [labels]\n\n";
            std::cout << "No image provided: using random values\n\n";
        }
        else if(argc == 4)
        {
            data_path = argv[2];
            image     = argv[3];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " [labels]\n\n";
            std::cout << "No text file with labels provided: skipping output accessor\n\n";
        }
        else
        {
            data_path = argv[2];
            image     = argv[3];
            label     = argv[4];
        }

        graph << target_hint
              << convolution_hint
              << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, 1U), 1, DataType::F32),
                        get_input_accessor(image, mean_r, mean_g, mean_b))
              // Layer 1
              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv1_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv1_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv1_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv1_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 2
              << ConvolutionLayer(
                  3U, 3U, 128U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv2_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv2_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 128U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv2_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv2_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 3
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_3_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_4_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv3_4_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 4
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_3_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_4_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv4_4_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 5
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_3_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_4_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/conv5_4_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 6
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/fc6_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/fc6_b.npy"))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 7
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/fc7_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/fc7_b.npy"))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 8
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/fc8_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg19_model/fc8_b.npy"))
              // Softmax
              << SoftmaxLayer()
              << Tensor(get_output_accessor(label, 5));
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    Graph graph{};
};

/** Main program for VGG19
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphVGG19Example>(argc, argv);
}
