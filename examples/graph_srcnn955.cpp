/*
 * Copyright (c) 2018-2019 ARM Limited.
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

/** Example demonstrating how to implement SRCNN 9-5-5 network using the Compute Library's graph API */
class GraphSRCNN955Example : public Example
{
public:
    GraphSRCNN955Example()
        : cmd_parser(), common_opts(cmd_parser), model_input_width(nullptr), model_input_height(nullptr), common_params(), graph(0, "SRCNN955")
    {
        model_input_width  = cmd_parser.add_option<SimpleOption<unsigned int>>("image-width", 300);
        model_input_height = cmd_parser.add_option<SimpleOption<unsigned int>>("image-height", 300);

        // Add model id option
        model_input_width->set_help("Input image width.");
        model_input_height->set_help("Input image height.");
    }
    GraphSRCNN955Example(const GraphSRCNN955Example &) = delete;
    GraphSRCNN955Example &operator=(const GraphSRCNN955Example &) = delete;
    GraphSRCNN955Example(GraphSRCNN955Example &&)                 = default; // NOLINT
    GraphSRCNN955Example &operator=(GraphSRCNN955Example &&) = default;      // NOLINT
    ~GraphSRCNN955Example() override                         = default;
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

        // Get input image width and height
        const unsigned int image_width  = model_input_width->value();
        const unsigned int image_height = model_input_height->value();

        // Print parameter values
        std::cout << common_params << std::endl;
        std::cout << "Image width: " << image_width << std::endl;
        std::cout << "Image height: " << image_height << std::endl;

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Get trainable parameters data path
        const std::string data_path  = common_params.data_path;
        const std::string model_path = "/cnn_data/srcnn955_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(image_width, image_height, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false /* Do not convert to BGR */))
              << ConvolutionLayer(
                  9U, 9U, 64U,
                  get_weights_accessor(data_path, "conv1_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "conv1_biases.npy"),
                  PadStrideInfo(1, 1, 4, 4))
              .set_name("conv1/convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv1/Relu")
              << ConvolutionLayer(
                  5U, 5U, 32U,
                  get_weights_accessor(data_path, "conv2_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "conv2_biases.npy"),
                  PadStrideInfo(1, 1, 2, 2))
              .set_name("conv2/convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/Relu")
              << ConvolutionLayer(
                  5U, 5U, 3U,
                  get_weights_accessor(data_path, "conv3_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "conv3_biases.npy"),
                  PadStrideInfo(1, 1, 2, 2))
              .set_name("conv3/convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv3/Relu")
              << OutputLayer(arm_compute::support::cpp14::make_unique<DummyAccessor>(0));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
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
    CommandLineParser           cmd_parser;
    CommonGraphOptions          common_opts;
    SimpleOption<unsigned int> *model_input_width{ nullptr };
    SimpleOption<unsigned int> *model_input_height{ nullptr };
    CommonGraphParams           common_params;
    Stream                      graph;
};

/** Main program for SRCNN 9-5-5
 *
 * Model is based on:
 *      http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
 *      "Image Super-Resolution Using Deep Convolutional Networks"
 *      Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphSRCNN955Example>(argc, argv);
}
