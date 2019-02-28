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
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement VGG based VDSR network using the Compute Library's graph API */
class GraphVDSRExample : public Example
{
public:
    GraphVDSRExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "VDSR")
    {
        model_input_width  = cmd_parser.add_option<SimpleOption<unsigned int>>("image-width", 192);
        model_input_height = cmd_parser.add_option<SimpleOption<unsigned int>>("image-height", 192);

        // Add model id option
        model_input_width->set_help("Input image width.");
        model_input_height->set_help("Input image height.");
    }
    GraphVDSRExample(const GraphVDSRExample &) = delete;
    GraphVDSRExample &operator=(const GraphVDSRExample &) = delete;
    GraphVDSRExample(GraphVDSRExample &&)                 = default; // NOLINT
    GraphVDSRExample &operator=(GraphVDSRExample &&) = default;      // NOLINT
    ~GraphVDSRExample() override                     = default;
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

        // Get trainable parameters data path
        const std::string data_path  = common_params.data_path;
        const std::string model_path = "/cnn_data/vdsr_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(image_width, image_height, 1U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        // Note: Quantization info are random and used only for benchmarking purposes
        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor.set_quantization_info(QuantizationInfo(0.0078125f, 128)),
                            get_input_accessor(common_params, std::move(preprocessor), false));

        SubStream left(graph);
        SubStream right(graph);

        // Layer 1
        right << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "conv0_w.npy", weights_layout),
                  get_weights_accessor(data_path, "conv0_b.npy"),
                  PadStrideInfo(1, 1, 1, 1), 1, QuantizationInfo(0.031778190285f, 156), QuantizationInfo(0.0784313753247f, 128))
              .set_name("conv0")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv0/Relu");

        // Rest 17 layers
        for(unsigned int i = 1; i < 19; ++i)
        {
            const std::string conv_w_path = "conv" + arm_compute::support::cpp11::to_string(i) + "_w.npy";
            const std::string conv_b_path = "conv" + arm_compute::support::cpp11::to_string(i) + "_b.npy";
            const std::string conv_name   = "conv" + arm_compute::support::cpp11::to_string(i);
            right << ConvolutionLayer(
                      3U, 3U, 64U,
                      get_weights_accessor(data_path, conv_w_path, weights_layout),
                      get_weights_accessor(data_path, conv_b_path),
                      PadStrideInfo(1, 1, 1, 1), 1, QuantizationInfo(0.015851572156f, 93))
                  .set_name(conv_name)
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(conv_name + "/Relu");
        }

        // Final layer
        right << ConvolutionLayer(
                  3U, 3U, 1U,
                  get_weights_accessor(data_path, "conv20_w.npy", weights_layout),
                  get_weights_accessor(data_path, "conv20_b.npy"),
                  PadStrideInfo(1, 1, 1, 1), 1, QuantizationInfo(0.015851572156f, 93))
              .set_name("conv20")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv20/Relu");

        // Add residual to input
        graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name("add")
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

/** Main program for VGG-based VDSR
 *
 * Model is based on:
 *      https://arxiv.org/pdf/1511.04587.pdf
 *      "Accurate Image Super-Resolution Using Very Deep Convolutional Networks"
 *      Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphVDSRExample>(argc, argv);
}
