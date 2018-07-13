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

/** Example demonstrating how to implement InceptionV4's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
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
        ARM_COMPUTE_ERROR_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "Unsupported data type!");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(TensorDescriptor(TensorShape(299U, 299U, 3U, 1U), common_params.data_type),
                            get_input_accessor(common_params, std::move(preprocessor), false))
              // Conv2d_1a_3x3
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_weights.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Conv2d_2a_3x3
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_weights.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Conv2d_2b_3x3
              << ConvolutionLayer(3U, 3U, 64U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_weights.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        graph << get_mixed_3a(data_path);
        graph << get_mixed_4a(data_path);
        graph << get_mixed_5a(data_path);
        // 4 inception A blocks
        graph << get_inceptionA_block(data_path, "Mixed_5b");
        graph << get_inceptionA_block(data_path, "Mixed_5c");
        graph << get_inceptionA_block(data_path, "Mixed_5d");
        graph << get_inceptionA_block(data_path, "Mixed_5e");
        // reduction A block
        graph << get_reductionA_block(data_path);
        // 7 inception B blocks
        graph << get_inceptionB_block(data_path, "Mixed_6b");
        graph << get_inceptionB_block(data_path, "Mixed_6c");
        graph << get_inceptionB_block(data_path, "Mixed_6d");
        graph << get_inceptionB_block(data_path, "Mixed_6e");
        graph << get_inceptionB_block(data_path, "Mixed_6f");
        graph << get_inceptionB_block(data_path, "Mixed_6g");
        graph << get_inceptionB_block(data_path, "Mixed_6h");
        // reduction B block
        graph << get_reductionB_block(data_path);
        // 3 inception C blocks
        graph << get_inceptionC_block(data_path, "Mixed_7b");
        graph << get_inceptionC_block(data_path, "Mixed_7c");
        graph << get_inceptionC_block(data_path, "Mixed_7d");
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << FlattenLayer()
              << FullyConnectedLayer(
                  1001U,
                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Logits_Logits_weights.npy"),
                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Logits_Logits_biases.npy"))
              << SoftmaxLayer()
              << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
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
    BranchLayer get_mixed_3a(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_3a_";

        SubStream i_a(graph);
        i_a << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }

    BranchLayer get_mixed_4a(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_4a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }

    BranchLayer get_mixed_5a(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_5a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(3U, 3U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }

    BranchLayer get_inceptionA_block(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(1U, 1U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    BranchLayer get_reductionA_block(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_6a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(3U, 3U, 384U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c));
    }

    BranchLayer get_inceptionB_block(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 384U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(1U, 1U, 128U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    BranchLayer get_reductionB_block(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_7a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 320U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 320U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c));
    }

    BranchLayer get_inceptionC_block(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b1(i_b);
        i_b1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b2(i_b);
        i_b2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // Merge b1 and b2
        i_b << BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_b1), std::move(i_b2));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                1U, 3U, 448U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 1))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 1U, 512U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c1(i_c);
        i_c1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c2(i_c);
        i_c2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // Merge i_c1 and i_c2
        i_c << BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_c1), std::move(i_c2));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Inception V4
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<InceptionV4Example>(argc, argv);
}
