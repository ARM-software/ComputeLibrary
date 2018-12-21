/*
 * Copyright (c) 2017-2018 ARM Limited.
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

/** Example demonstrating how to implement InceptionV3's network using the Compute Library's graph API */
class InceptionV3Example : public Example
{
public:
    InceptionV3Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "InceptionV3")
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
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(299U, 299U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false))
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
              .set_name("Conv2d_1a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv3_model/Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_1a_3x3/BatchNorm/batchnorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_1a_3x3/Relu")
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_2a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv3_model/Conv2d_2a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_2a_3x3/BatchNorm/batchnorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_2a_3x3/Relu")

              << ConvolutionLayer(3U, 3U, 64U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
              .set_name("Conv2d_2b_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv3_model/Conv2d_2b_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_2b_3x3/BatchNorm/batchnorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_2b_3x3/Relu")

              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("MaxPool_3a_3x3/MaxPool")

              << ConvolutionLayer(1U, 1U, 80U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_3b_1x1/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv3_model/Conv2d_3b_1x1_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_3b_1x1/BatchNorm/batchnorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_3b_1x1/Relu")

              << ConvolutionLayer(3U, 3U, 192U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_4a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv3_model/Conv2d_4a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_4a_3x3/BatchNorm/batchnorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_4a_3x3/Relu")

              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("MaxPool_5a_3x3/MaxPool");

        graph << get_inception_node_A(data_path, "Mixed_5b", weights_layout, 64U, std::make_tuple(48U, 64U), std::make_tuple(64U, 96U, 96U),
                                      32U)
              .set_name("Mixed_5b/concat");
        graph << get_inception_node_A(data_path, "Mixed_5c", weights_layout, 64U, std::make_tuple(48U, 64U), std::make_tuple(64U, 96U, 96U),
                                      64U, true)
              .set_name("Mixed_5c/concat");
        graph << get_inception_node_A(data_path, "Mixed_5d", weights_layout, 64U, std::make_tuple(48U, 64U), std::make_tuple(64U, 96U, 96U),
                                      64U)
              .set_name("Mixed_5d/concat");

        graph << get_inception_node_B(data_path, "Mixed_6a", weights_layout, 384U, std::make_tuple(64U, 96U, 96U)).set_name("Mixed_6a/concat");

        graph << get_inception_node_C(data_path, "Mixed_6b", weights_layout, 192U, std::make_tuple(128U, 128U, 192U),
                                      std::make_tuple(128U, 128U, 128U, 128U, 192U), 192U)
              .set_name("Mixed_6b/concat");
        graph << get_inception_node_C(data_path, "Mixed_6c", weights_layout, 192U, std::make_tuple(160U, 160U, 192U),
                                      std::make_tuple(160U, 160U, 160U, 160U, 192U), 192U)
              .set_name("Mixed_6c/concat");
        graph << get_inception_node_C(data_path, "Mixed_6d", weights_layout, 192U, std::make_tuple(160U, 160U, 192U),
                                      std::make_tuple(160U, 160U, 160U, 160U, 192U), 192U)
              .set_name("Mixed_6d/concat");
        graph << get_inception_node_C(data_path, "Mixed_6e", weights_layout, 192U, std::make_tuple(192U, 192U, 192U),
                                      std::make_tuple(192U, 192U, 192U, 192U, 192U), 192U)
              .set_name("Mixed_6e/concat");

        graph << get_inception_node_D(data_path, "Mixed_7a", weights_layout, std::make_tuple(192U, 320U),
                                      std::make_tuple(192U, 192U, 192U, 192U))
              .set_name("Mixed_7a/concat");

        graph << get_inception_node_E(data_path, "Mixed_7b", weights_layout, 320U, std::make_tuple(384U, 384U, 384U),
                                      std::make_tuple(448U, 384U, 384U, 384U), 192U)
              .set_name("Mixed_7b/concat");
        graph << get_inception_node_E(data_path, "Mixed_7c", weights_layout, 320U, std::make_tuple(384U, 384U, 384U),
                                      std::make_tuple(448U, 384U, 384U, 384U), 192U, true)
              .set_name("Mixed_7c/concat");

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 8, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL))).set_name("Logits/AvgPool_1a_8x8/AvgPool")
              << ConvolutionLayer(1U, 1U, 1001U, get_weights_accessor(data_path,
                                                                      "/cnn_data/inceptionv3_model/Logits_Conv2d_1c_1x1_weights.npy", weights_layout),
                                  get_weights_accessor(data_path,
                                                       "/cnn_data/inceptionv3_model/Logits_Conv2d_1c_1x1_biases.npy"),
                                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Logits/Conv2d_1c_1x1/convolution")
              << ReshapeLayer(TensorShape(1001U)).set_name("Predictions/Reshape")
              << SoftmaxLayer().set_name("Predictions/Softmax")
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
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

private:
    ConcatLayer get_inception_node_A(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int a_filt,
                                     std::tuple<unsigned int, unsigned int> b_filters,
                                     std::tuple<unsigned int, unsigned int, unsigned int> c_filters,
                                     unsigned int d_filt,
                                     bool         is_name_different = false)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";

        // This is due to a naming issue in the tf model
        std::string conv_id0 = "_0a_";
        std::string conv_id1 = "2d_0b_";
        if(is_name_different)
        {
            conv_id0 = "_0b_";
            conv_id1 = "_1_0c_";
        }

        SubStream i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 + "1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d" + conv_id0 + "1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 + "1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 + "1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id0 + "1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d" + conv_id0 + "1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d" + conv_id0 + "1x1/Relu")
            << ConvolutionLayer(
                5U, 5U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv" + conv_id1 + "5x5_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 2, 2))
            .set_name(param_path + "/Branch_1/Conv2d" + conv_id1 + "5x5/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv" + conv_id1 + "5x5_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv" + conv_id1 + "5x5_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv" + conv_id1 + "5x5_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d" + conv_id1 + "5x5/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d" + conv_id1 + "5x5/Relu");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, std::get<0>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                3U, 3U, std::get<1>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(
                3U, 3U, std::get<2>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/Branch_2/Conv2d_0c_3x3/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0c_3x3/BatchNorm/batcnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0c_3x3/Relu");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true)).set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(
                1U, 1U, d_filt,
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    ConcatLayer get_inception_node_B(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int a_filt,
                                     std::tuple<unsigned int, unsigned int, unsigned int> b_filters)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                3U, 3U, a_filt,
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(2, 2, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_1a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_1a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_1a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                3U, 3U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/Branch_1/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0b_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(
                3U, 3U, std::get<2>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(2, 2, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_1a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_1a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_1a_1x1/Relu");

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name(param_path + "/Branch_2/MaxPool_1a_3x3/MaxPool");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c));
    }

    ConcatLayer get_inception_node_C(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int a_filt,
                                     std::tuple<unsigned int, unsigned int, unsigned int> b_filters,
                                     std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int> c_filters,
                                     unsigned int d_filt)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                7U, 1U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 3, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0b_1x7/Relu")
            << ConvolutionLayer(
                1U, 7U, std::get<2>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 3))
            .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0c_7x1/Relu");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, std::get<0>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                1U, 7U, std::get<1>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 3))
            .set_name(param_path + "/Branch_2/Conv2d_0b_7x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0b_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0b_7x1/Relu")
            << ConvolutionLayer(
                7U, 1U, std::get<2>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 3, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0c_1x7/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0c_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0c_1x7/Relu")
            << ConvolutionLayer(
                1U, 7U, std::get<3>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 3))
            .set_name(param_path + "/Branch_2/Conv2d_0d_7x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0d_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0d_7x1/Relu")
            << ConvolutionLayer(
                7U, 1U, std::get<4>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 3, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0e_1x7/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0e_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0e_1x7/Relu");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true)).set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(
                1U, 1U, d_filt,
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    ConcatLayer get_inception_node_D(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     std::tuple<unsigned int, unsigned int> a_filters,
                                     std::tuple<unsigned int, unsigned int, unsigned int, unsigned int> b_filters)
    {
        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, std::get<0>(a_filters),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                3U, 3U, std::get<1>(a_filters),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(2, 2, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_1a_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_1a_3x3/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                7U, 1U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 3, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0b_1x7/Relu")
            << ConvolutionLayer(
                1U, 7U, std::get<2>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 3))
            .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0c_7x1/Relu")
            << ConvolutionLayer(
                3U, 3U, std::get<3>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(2, 2, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_1a_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_1a_3x3/Relu");

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name(param_path + "/Branch_2/MaxPool_1a_3x3/MaxPool");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c));
    }

    ConcatLayer get_inception_node_E(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int a_filt,
                                     std::tuple<unsigned int, unsigned int, unsigned int> b_filters,
                                     std::tuple<unsigned int, unsigned int, unsigned int, unsigned int> c_filters,
                                     unsigned int d_filt,
                                     bool         is_name_different = false)
    {
        // This is due to a naming issue in the tf model
        std::string conv_id = "_0b_";
        if(is_name_different)
        {
            conv_id = "_0c_";
        }

        std::string total_path = "/cnn_data/inceptionv3_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu");

        SubStream i_b1(i_b);
        i_b1 << ConvolutionLayer(
                 3U, 1U, std::get<1>(b_filters),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             .set_name(param_path + "/Branch_1/Conv2d_0b_1x3/convolution")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_1/Conv2d_0b_1x3/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0b_1x3/Relu");

        SubStream i_b2(i_b);
        i_b2 << ConvolutionLayer(
                 1U, 3U, std::get<2>(b_filters),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id + "3x1_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             .set_name(param_path + "/Branch_1/Conv2d" + conv_id + "3x1/convolution")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id + "3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id + "3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d" + conv_id + "3x1_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_1/Conv2d" + conv_id + "3x1/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d" + conv_id + "3x1/Relu");

        // Merge b1 and b2
        i_b << ConcatLayer(std::move(i_b1), std::move(i_b2)).set_name(param_path + "/Branch_1/concat");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, std::get<0>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                3U, 3U, std::get<1>(c_filters),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0b_3x3/Relu");

        SubStream i_c1(i_c);
        i_c1 << ConvolutionLayer(
                 3U, 1U, std::get<2>(c_filters),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             .set_name(param_path + "/Branch_2/Conv2d_0c_1x3/convolution")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_2/Conv2d_0c_1x3/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0c_1x3/Relu");

        SubStream i_c2(i_c);
        i_c2 << ConvolutionLayer(
                 1U, 3U, std::get<3>(c_filters),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_3x1_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             .set_name(param_path + "/Branch_2/Conv2d_0d_3x1/convolution")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_3x1_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_2/Conv2d_0d_3x1/BatchNorm/batchnorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0d_3x1/Relu");

        // Merge i_c1 and i_c2
        i_c << ConcatLayer(std::move(i_c1), std::move(i_c2)).set_name(param_path + "/Branch_2/concat");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true)).set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(
                1U, 1U, d_filt,
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/convolution")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm/batchnorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Inception V3
 *
 * Model is based on:
 *      https://arxiv.org/abs/1512.00567
 *      "Rethinking the Inception Architecture for Computer Vision"
 *      Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
 *
 * Provenance: download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<InceptionV3Example>(argc, argv);
}
