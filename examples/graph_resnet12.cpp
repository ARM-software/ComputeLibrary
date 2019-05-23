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

/** Example demonstrating how to implement ResNet12 network using the Compute Library's graph API */
class GraphResNet12Example : public Example
{
public:
    GraphResNet12Example()
        : cmd_parser(), common_opts(cmd_parser), model_input_width(nullptr), model_input_height(nullptr), common_params(), graph(0, "ResNet12")
    {
        model_input_width  = cmd_parser.add_option<SimpleOption<unsigned int>>("image-width", 192);
        model_input_height = cmd_parser.add_option<SimpleOption<unsigned int>>("image-height", 128);

        // Add model id option
        model_input_width->set_help("Input image width.");
        model_input_height->set_help("Input image height.");
    }
    GraphResNet12Example(const GraphResNet12Example &) = delete;
    GraphResNet12Example &operator=(const GraphResNet12Example &) = delete;
    GraphResNet12Example(GraphResNet12Example &&)                 = default; // NOLINT
    GraphResNet12Example &operator=(GraphResNet12Example &&) = default;      // NOLINT
    ~GraphResNet12Example() override                         = default;
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

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;
        std::cout << "Image width: " << image_width << std::endl;
        std::cout << "Image height: " << image_height << std::endl;

        // Get trainable parameters data path
        const std::string data_path  = common_params.data_path;
        const std::string model_path = "/cnn_data/resnet12_model/";

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
                  get_weights_accessor(data_path, "conv1_biases.npy", weights_layout),
                  PadStrideInfo(1, 1, 4, 4))
              .set_name("conv1/convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv1/Relu");

        add_residual_block(data_path, "block1", weights_layout);
        add_residual_block(data_path, "block2", weights_layout);
        add_residual_block(data_path, "block3", weights_layout);
        add_residual_block(data_path, "block4", weights_layout);

        graph << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "conv10_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "conv10_biases.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv10/convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv10/Relu")
              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "conv11_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "conv11_biases.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv11/convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv11/Relu")
              << ConvolutionLayer(
                  9U, 9U, 3U,
                  get_weights_accessor(data_path, "conv12_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "conv12_biases.npy"),
                  PadStrideInfo(1, 1, 4, 4))
              .set_name("conv12/convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH)).set_name("conv12/Tanh")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 0.58f, 0.5f)).set_name("conv12/Linear")
              << OutputLayer(arm_compute::support::cpp14::make_unique<DummyAccessor>(0));

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
    CommandLineParser           cmd_parser;
    CommonGraphOptions          common_opts;
    SimpleOption<unsigned int> *model_input_width{ nullptr };
    SimpleOption<unsigned int> *model_input_height{ nullptr };
    CommonGraphParams           common_params;
    Stream                      graph;

    void add_residual_block(const std::string &data_path, const std::string &name, DataLayout weights_layout)
    {
        std::stringstream unit_path_ss;
        unit_path_ss << data_path << name << "_";
        std::stringstream unit_name_ss;
        unit_name_ss << name << "/";

        std::string unit_path = unit_path_ss.str();
        std::string unit_name = unit_name_ss.str();

        SubStream left(graph);
        SubStream right(graph);

        right << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, unit_path + "conv1_weights.npy", weights_layout),
                  get_weights_accessor(data_path, unit_path + "conv1_biases.npy", weights_layout),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name(unit_name + "conv1/convolution")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_beta.npy"),
                  0.0000100099996416f)
              .set_name(unit_name + "conv1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, unit_path + "conv2_weights.npy", weights_layout),
                  get_weights_accessor(data_path, unit_path + "conv2_biases.npy", weights_layout),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name(unit_name + "conv2/convolution")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_beta.npy"),
                  0.0000100099996416f)
              .set_name(unit_name + "conv2/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv2/Relu");

        graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
    }
};

/** Main program for ResNet12
 *
 * Model is based on:
 *      https://arxiv.org/pdf/1709.01118.pdf
 *      "WESPE: Weakly Supervised Photo Enhancer for Digital Cameras"
 *      Andrey Ignatov, Nikolay Kobyshev, Kenneth Vanhoey, Radu Timofte, Luc Van Gool
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphResNet12Example>(argc, argv);
}
