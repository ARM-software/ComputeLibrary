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

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement Squeezenet's v1.1 network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
class GraphSqueezenet_v1_1Example : public Example
{
public:
    GraphSqueezenet_v1_1Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "SqueezeNetV1.1")
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
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "Unsupported data type!");
        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_layout == DataLayout::NHWC, "Unsupported data layout!");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(TensorDescriptor(TensorShape(227U, 227U, 3U, 1U), common_params.data_type),
                            get_input_accessor(common_params, std::move(preprocessor)))
              << ConvolutionMethod::DIRECT
              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/conv1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/conv1_b.npy"),
                  PadStrideInfo(2, 2, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              << ConvolutionMethod::DEFAULT
              << ConvolutionLayer(
                  1U, 1U, 16U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire2_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire2_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire2", 64U, 64U);
        graph << ConvolutionLayer(
                  1U, 1U, 16U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire3_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire3_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire3", 64U, 64U);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              << ConvolutionLayer(
                  1U, 1U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire4_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire4_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire4", 128U, 128U);
        graph << ConvolutionLayer(
                  1U, 1U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire5_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire5_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire5", 128U, 128U);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              << ConvolutionLayer(
                  1U, 1U, 48U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire6_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire6_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire6", 192U, 192U);
        graph << ConvolutionLayer(
                  1U, 1U, 48U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire7_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire7_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire7", 192U, 192U);
        graph << ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire8_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire8_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire8", 256U, 256U);
        graph << ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire9_squeeze1x1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/fire9_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire9", 256U, 256U);
        graph << ConvolutionLayer(
                  1U, 1U, 1000U,
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/conv10_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/squeezenet_v1_1_model/conv10_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << FlattenLayer()
              << SoftmaxLayer()
              << OutputLayer(get_output_accessor(common_params, 5));

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
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    BranchLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, unsigned int expand1_filt, unsigned int expand3_filt)
    {
        std::string total_path = "/cnn_data/squeezenet_v1_1_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, expand1_filt,
                get_weights_accessor(data_path, total_path + "expand1x1_w.npy"),
                get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                3U, 3U, expand3_filt,
                get_weights_accessor(data_path, total_path + "expand3x3_w.npy"),
                get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }
};

/** Main program for Squeezenet v1.1
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphSqueezenet_v1_1Example>(argc, argv);
}
