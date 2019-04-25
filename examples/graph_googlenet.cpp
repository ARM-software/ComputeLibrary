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

/** Example demonstrating how to implement Googlenet's network using the Compute Library's graph API */
class GraphGooglenetExample : public Example
{
public:
    GraphGooglenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GoogleNet")
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
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
              << ConvolutionLayer(
                  7U, 7U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_b.npy"),
                  PadStrideInfo(2, 2, 3, 3))
              .set_name("conv1/7x7_s2")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv1/relu_7x7")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool1/3x3_s2")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("pool1/norm1")
              << ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv2/3x3_reduce")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3_reduce")
              << ConvolutionLayer(
                  3U, 3U, 192U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv2/3x3")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv2/relu_3x3")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("conv2/norm2")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool2/3x3_s2");
        graph << get_inception_node(data_path, "inception_3a", weights_layout, 64, std::make_tuple(96U, 128U), std::make_tuple(16U, 32U), 32U).set_name("inception_3a/concat");
        graph << get_inception_node(data_path, "inception_3b", weights_layout, 128, std::make_tuple(128U, 192U), std::make_tuple(32U, 96U), 64U).set_name("inception_3b/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool3/3x3_s2");
        graph << get_inception_node(data_path, "inception_4a", weights_layout, 192, std::make_tuple(96U, 208U), std::make_tuple(16U, 48U), 64U).set_name("inception_4a/concat");
        graph << get_inception_node(data_path, "inception_4b", weights_layout, 160, std::make_tuple(112U, 224U), std::make_tuple(24U, 64U), 64U).set_name("inception_4b/concat");
        graph << get_inception_node(data_path, "inception_4c", weights_layout, 128, std::make_tuple(128U, 256U), std::make_tuple(24U, 64U), 64U).set_name("inception_4c/concat");
        graph << get_inception_node(data_path, "inception_4d", weights_layout, 112, std::make_tuple(144U, 288U), std::make_tuple(32U, 64U), 64U).set_name("inception_4d/concat");
        graph << get_inception_node(data_path, "inception_4e", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_4e/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4/3x3_s2");
        graph << get_inception_node(data_path, "inception_5a", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U).set_name("inception_5a/concat");
        graph << get_inception_node(data_path, "inception_5b", weights_layout, 384, std::make_tuple(192U, 384U), std::make_tuple(48U, 128U), 128U).set_name("inception_5b/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL))).set_name("pool5/7x7_s1")
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_b.npy"))
              .set_name("loss3/classifier")
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

    ConcatLayer get_inception_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                   unsigned int a_filt,
                                   std::tuple<unsigned int, unsigned int> b_filters,
                                   std::tuple<unsigned int, unsigned int> c_filters,
                                   unsigned int d_filt)
    {
        std::string total_path = "/cnn_data/googlenet_model/" + param_path + "/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/1x1")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_1x1");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_reduce_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "3x3_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/3x3_reduce")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_3x3_reduce")
            << ConvolutionLayer(
                3U, 3U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/3x3")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_3x3");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, std::get<0>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_reduce_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "5x5_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/5x5_reduce")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_5x5_reduce")
            << ConvolutionLayer(
                5U, 5U, std::get<1>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "5x5_b.npy"),
                PadStrideInfo(1, 1, 2, 2))
            .set_name(param_path + "/5x5")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_5x5");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL))).set_name(param_path + "/pool")
            << ConvolutionLayer(
                1U, 1U, d_filt,
                get_weights_accessor(data_path, total_path + "pool_proj_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "pool_proj_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/pool_proj")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/relu_pool_proj");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Googlenet
 *
 * Model is based on:
 *      https://arxiv.org/abs/1409.4842
 *      "Going deeper with convolutions"
 *      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
 *
 * Provenance: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphGooglenetExample>(argc, argv);
}
