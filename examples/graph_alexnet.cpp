/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement AlexNet's network using the Compute Library's graph API */
class GraphAlexnetExample : public Example
{
public:
    GraphAlexnetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "AlexNet")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(227U, 227U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
              // Layer 1
              << ConvolutionLayer(
                  11U, 11U, 96U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
                  PadStrideInfo(4, 4, 0, 0))
              .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool1")
              // Layer 2
              << ConvolutionLayer(
                  5U, 5U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"),
                  PadStrideInfo(1, 1, 2, 2), 2)
              .set_name("conv2")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm2")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool2")
              // Layer 3
              << ConvolutionLayer(
                  3U, 3U, 384U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv3")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3")
              // Layer 4
              << ConvolutionLayer(
                  3U, 3U, 384U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
                  PadStrideInfo(1, 1, 1, 1), 2)
              .set_name("conv4")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4")
              // Layer 5
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
                  PadStrideInfo(1, 1, 1, 1), 2)
              .set_name("conv5")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool5")
              // Layer 6
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
              .set_name("fc6")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu6")
              // Layer 7
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
              .set_name("fc7")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu7")
              // Layer 8
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
              .set_name("fc8")
              // Softmax
              << SoftmaxLayer().set_name("prob")
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;

        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
};

/** Main program for AlexNet
 *
 * Model is based on:
 *      https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
 *      "ImageNet Classification with Deep Convolutional Neural Networks"
 *      Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E
 *
 * Provenance: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 * @return Return code
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphAlexnetExample>(argc, argv);
}
