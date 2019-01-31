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

/** Example demonstrating how to implement InceptionV4's network using the Compute Library's graph API */
class InceptionV4Example final : public Example
{
public:
    InceptionV4Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "InceptionV4")
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
              // Conv2d_1a_3x3
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
              .set_name("Conv2d_1a_3x3/Conv2D")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_1a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_1a_3x3/Relu")
              // Conv2d_2a_3x3
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_2a_3x3/Conv2D")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_2a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_2a_3x3/Relu")
              // Conv2d_2b_3x3
              << ConvolutionLayer(3U, 3U, 64U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
              .set_name("Conv2d_2b_3x3/Conv2D")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              .set_name("Conv2d_2b_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_2b_3x3/Relu");

        graph << get_mixed_3a(data_path, weights_layout).set_name("Mixed_3a/concat");
        graph << get_mixed_4a(data_path, weights_layout).set_name("Mixed_4a/concat");
        graph << get_mixed_5a(data_path, weights_layout).set_name("Mixed_5a/concat");
        // 4 inception A blocks
        graph << get_inceptionA_block(data_path, weights_layout, "Mixed_5b").set_name("Mixed_5b/concat");
        graph << get_inceptionA_block(data_path, weights_layout, "Mixed_5c").set_name("Mixed_5c/concat");
        graph << get_inceptionA_block(data_path, weights_layout, "Mixed_5d").set_name("Mixed_5d/concat");
        graph << get_inceptionA_block(data_path, weights_layout, "Mixed_5e").set_name("Mixed_5e/concat");
        // reduction A block
        graph << get_reductionA_block(data_path, weights_layout).set_name("Mixed_6a/concat");
        // 7 inception B blocks
        graph << get_inceptionB_block(data_path, weights_layout, "Mixed_6b").set_name("Mixed_6b/concat");
        graph << get_inceptionB_block(data_path, weights_layout, "Mixed_6c").set_name("Mixed_6c/concat");
        graph << get_inceptionB_block(data_path, weights_layout, "Mixed_6d").set_name("Mixed_6d/concat");
        graph << get_inceptionB_block(data_path, weights_layout, "Mixed_6e").set_name("Mixed_6e/concat");
        graph << get_inceptionB_block(data_path, weights_layout, "Mixed_6f").set_name("Mixed_6f/concat");
        graph << get_inceptionB_block(data_path, weights_layout, "Mixed_6g").set_name("Mixed_6g/concat");
        graph << get_inceptionB_block(data_path, weights_layout, "Mixed_6h").set_name("Mixed_6h/concat");
        // reduction B block
        graph << get_reductionB_block(data_path, weights_layout).set_name("Mixed_7a/concat");
        // 3 inception C blocks
        graph << get_inceptionC_block(data_path, weights_layout, "Mixed_7b").set_name("Mixed_7b/concat");
        graph << get_inceptionC_block(data_path, weights_layout, "Mixed_7c").set_name("Mixed_7c/concat");
        graph << get_inceptionC_block(data_path, weights_layout, "Mixed_7d").set_name("Mixed_7d/concat");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("Logits/AvgPool_1a/AvgPool")
              << FlattenLayer().set_name("Logits/Flatten")
              << FullyConnectedLayer(
                  1001U,
                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Logits_Logits_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Logits_Logits_biases.npy"))
              .set_name("Logits/MatMul")
              << SoftmaxLayer().set_name("Logits/Predictions")
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
    ConcatLayer get_mixed_3a(const std::string &data_path, DataLayout weights_layout)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_3a_";

        SubStream i_a(graph);
        i_a << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true)).set_name("Mixed_3a/Branch_0/MaxPool_0a_3x3/MaxPool");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_3a/Branch_1/Conv2d_0a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_3a/Branch_1/Conv2d_0a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_3a/Branch_1/Conv2d_0a_3x3/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b));
    }

    ConcatLayer get_mixed_4a(const std::string &data_path, DataLayout weights_layout)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_4a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_4a/Branch_0/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_4a/Branch_0/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_4a/Branch_0/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_4a/Branch_0/Conv2d_1a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_4a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_4a/Branch_0/Conv2d_1a_3x3/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_4a/Branch_1/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_4a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_4a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(7U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            .set_name("Mixed_4a/Branch_1/Conv2d_0b_1x7/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_4a/Branch_1/Conv2d_0b_1x7/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_4a/Branch_1/Conv2d_0b_1x7/Relu")
            << ConvolutionLayer(1U, 7U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            .set_name("Mixed_4a/Branch_1/Conv2d_0c_7x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_4a/Branch_1/Conv2d_0c_7x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_4a/Branch_1/Conv2d_0c_7x1/Relu")
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_4a/Branch_1/Conv2d_1a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_4a/Branch_1/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_4a/Branch_1/Conv2d_1a_3x3/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b));
    }

    ConcatLayer get_mixed_5a(const std::string &data_path, DataLayout weights_layout)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_5a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(3U, 3U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_5a/Branch_0/Conv2d_1a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_5a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5a/Branch_0/Conv2d_1a_3x3/Relu");

        SubStream i_b(graph);
        i_b << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true)).set_name("Mixed_5a/Branch_1/MaxPool_1a_3x3/MaxPool");

        return ConcatLayer(std::move(i_a), std::move(i_b));
    }

    ConcatLayer get_inceptionA_block(const std::string &data_path, DataLayout weights_layout, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/Branch_1/Conv2d_0b_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0b_3x3/Relu");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            .set_name(param_path + "/Branch_2/Conv2d_0c_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0c_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0c_3x3/Relu");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true)).set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(1U, 1U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    ConcatLayer get_reductionA_block(const std::string &data_path, DataLayout weights_layout)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_6a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(3U, 3U, 384U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            .set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(3U, 3U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/Relu");

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true)).set_name("Mixed_6a/Branch_2/MaxPool_1a_3x3/MaxPool");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c));
    }

    ConcatLayer get_inceptionB_block(const std::string &data_path, DataLayout weights_layout, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 384U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(7U, 1U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0b_1x7/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0b_1x7/Relu")
            << ConvolutionLayer(1U, 7U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0c_7x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0c_7x1/Relu");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(1U, 7U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            .set_name(param_path + "/Branch_2/Conv2d_0b_7x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0b_7x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0b_7x1/Relu")
            << ConvolutionLayer(7U, 1U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0c_1x7/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0c_1x7/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0c_1x7/Relu")
            << ConvolutionLayer(1U, 7U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            .set_name(param_path + "/Branch_2/Conv2d_0d_7x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0d_7x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0d_7x1/Relu")
            << ConvolutionLayer(7U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0e_1x7/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0e_1x7/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0e_1x7/Relu");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true)).set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(1U, 1U, 128U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    ConcatLayer get_reductionB_block(const std::string &data_path, DataLayout weights_layout)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_7a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(7U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            .set_name("Mixed_7a/Branch_1/Conv2d_0b_1x7/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_7a/Branch_1/Conv2d_0b_1x7/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_1/Conv2d_0b_1x7/Relu")
            << ConvolutionLayer(1U, 7U, 320U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            .set_name("Mixed_7a/Branch_1/Conv2d_0c_7x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_7a/Branch_1/Conv2d_0c_7x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_1/Conv2d_0c_7x1/Relu")
            << ConvolutionLayer(3U, 3U, 320U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/Relu");

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true)).set_name("Mixed_7a/Branch_2/MaxPool_1a_3x3/MaxPool");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c));
    }

    ConcatLayer get_inceptionC_block(const std::string &data_path, DataLayout weights_layout, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_0/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_0/Conv2d_0a_1x1/Relu");

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0a_1x1/Relu");

        SubStream i_b1(i_b);
        i_b1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             .set_name(param_path + "/Branch_1/Conv2d_0b_1x3/Conv2D")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_1/Conv2d_0b_1x3/BatchNorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0b_1x3/Relu");

        SubStream i_b2(i_b);
        i_b2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             .set_name(param_path + "/Branch_1/Conv2d_0c_3x1/Conv2D")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_1/Conv2d_0c_3x1/BatchNorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_1/Conv2d_0c_3x1/Relu");

        // Merge b1 and b2
        i_b << ConcatLayer(std::move(i_b1), std::move(i_b2)).set_name(param_path + "/Branch_1/concat");

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Conv2D")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                1U, 3U, 448U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 1))
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x1/Conv2D")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0b_3x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0b_3x1/Relu")
            << ConvolutionLayer(
                3U, 1U, 512U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_weights.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 0))
            .set_name(param_path + "/Branch_2/Conv2d_0c_1x3/Conv2D")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_beta.npy"),
                0.001f)
            .set_name(param_path + "/Branch_2/Conv2d_0c_1x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0c_1x3/Relu");

        SubStream i_c1(i_c);
        i_c1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             .set_name(param_path + "/Branch_2/Conv2d_0d_1x3/Conv2D")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_2/Conv2d_0d_1x3/BatchNorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0d_1x3/Relu");

        SubStream i_c2(i_c);
        i_c2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_weights.npy", weights_layout),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             .set_name(param_path + "/Branch_2/Conv2d_0e_3x1/Conv2D")
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_beta.npy"),
                 0.001f)
             .set_name(param_path + "/Branch_2/Conv2d_0e_3x1/BatchNorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_2/Conv2d_0e_3x1/Relu");

        // Merge i_c1 and i_c2
        i_c << ConcatLayer(std::move(i_c1), std::move(i_c2)).set_name(param_path + "/Branch_2/concat");

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true)).set_name(param_path + "/Branch_3/AvgPool_0a_3x3/AvgPool")
            << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Conv2D")
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            .set_name(param_path + "/Branch_3/Conv2d_0b_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "/Branch_3/Conv2d_0b_1x1/Relu");

        return ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Inception V4
 *
 * Model is based on:
 *      https://arxiv.org/abs/1602.07261
 *      "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
 *      Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
 *
 * Provenance: download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<InceptionV4Example>(argc, argv);
}
