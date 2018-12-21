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

/** Example demonstrating how to implement InceptionV4's network using the Compute Library's graph API */
class InceptionResNetV2Example final : public Example
{
public:
    InceptionResNetV2Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "InceptionResNetV2")
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

        // Set default layout if needed
        if(!common_opts.data_layout->is_set() && common_params.target == Target::NEON)
        {
            common_params.data_layout = DataLayout::NCHW;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Create model path
        std::string data_path  = common_params.data_path;
        std::string model_path = "/cnn_data/inception_resnet_v2_model/";
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>(0.f, 1.f);

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
                                  get_weights_accessor(data_path, "Conv2d_1a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(2, 2, 0, 0))
              .set_name("Conv2d_1a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv2d_1a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_1a_3x3/Relu")
              // Conv2d_2a_3x3
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "Conv2d_2a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_2a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_2a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_2a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_2a_3x3_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv2d_2a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_2a_3x3/Relu")
              // Conv2d_2b_3x3
              << ConvolutionLayer(3U, 3U, 64U,
                                  get_weights_accessor(data_path, "Conv2d_2b_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(1, 1, 1, 1))
              .set_name("Conv2d_2b_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_2b_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_2b_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_2b_3x3_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv2d_2b_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_2b_3x3/Relu")
              // MaxPool_3a_3x3
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true)).set_name("MaxPool_3a_3x3/MaxPool")
              // Conv2d_3b_1x1
              << ConvolutionLayer(1U, 1U, 80U,
                                  get_weights_accessor(data_path, "Conv2d_3b_1x1_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_3b_1x1/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_3b_1x1_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_3b_1x1_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_3b_1x1_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv2d_3b_1x1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_3b_1x1/Relu")
              // Conv2d_4a_3x3
              << ConvolutionLayer(3U, 3U, 192U,
                                  get_weights_accessor(data_path, "Conv2d_4a_3x3_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_4a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_4a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_4a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_4a_3x3_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv2d_4a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_4a_3x3/Relu")
              // MaxPool_5a_3x3
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0), true)).set_name("MaxPool_5a_3x3/MaxPool");

        block_mixed_5b(data_path, weights_layout);
        block35_repeat(data_path, weights_layout, 10);
        block_mixed_6a(data_path, weights_layout);
        block17_repeat(data_path, weights_layout, 20);
        block_mixed_7a(data_path, weights_layout);
        block8_repeat(data_path, weights_layout, 9, 0.2f, true);
        block8_repeat(data_path, weights_layout, 1, 1.f, false);

        // Conv2d_7b_1x1
        graph << ConvolutionLayer(1U, 1U, 1536U,
                                  get_weights_accessor(data_path, "Conv2d_7b_1x1_weights.npy", weights_layout),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv2d_7b_1x1/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_7b_1x1_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_7b_1x1_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_7b_1x1_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv2d_7b_1x1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv2d_7b_1x1/Relu")
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("Logits/AvgPool_1a_8x8")
              << FlattenLayer().set_name("Logits/Flatten")
              << FullyConnectedLayer(
                  1001U,
                  get_weights_accessor(data_path, "Logits_Logits_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "Logits_Logits_biases.npy"))
              .set_name("Logits/Logits")
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
    void block_mixed_5b(const std::string &data_path, DataLayout weights_layout)
    {
        // Branch 0
        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 96U,
                                get_weights_accessor(data_path, "Mixed_5b_Branch_0_Conv2d_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_5b/Branch_0/Conv2d_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_5b_Branch_0_Conv2d_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_0_Conv2d_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_0_Conv2d_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_5b/Branch_0/Conv2d_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5b/Branch_0/Conv2d_1x1/Relu");

        // Branch 1
        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 48U,
                                get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_5b/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5b/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(5U, 5U, 64U,
                                get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0b_5x5_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 2, 2))
            .set_name("Mixed_5b/Branch_1/Conv2d_0b_5x5/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0b_5x5_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0b_5x5_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_1_Conv2d_0b_5x5_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_5b/Branch_1/Conv2d_0b_5x5/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5b/Branch_1/Conv2d_0b_5x5/Relu");

        // Branch 2
        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_5b/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5b/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 1, 1))
            .set_name("Mixed_5b/Branch_2/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_5b/Branch_2/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5b/Branch_2/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0c_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 1, 1))
            .set_name("Mixed_5b/Branch_2/Conv2d_0c_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_5b/Branch_2/Conv2d_0c_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5b/Branch_2/Conv2d_0c_3x3/Relu");

        // Branch 3
        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true)).set_name("Mixed_5b/Branch_3/AvgPool_0a_3x3")
            << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, "Mixed_5b_Branch_3_Conv2d_0b_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_5b/Branch_3/Conv2d_0b_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_5b/Branch_3/Conv2d_0b_1x1/Relu");

        // Concatenate
        graph << ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d)).set_name("Mixed_5a/concat");
    }

    void block_mixed_6a(const std::string &data_path, DataLayout weights_layout)
    {
        // Branch 0
        SubStream i_a(graph);
        i_a << ConvolutionLayer(3U, 3U, 384U,
                                get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/Relu");

        // Branch 1
        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 256U,
                                get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 1, 1))
            .set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(3U, 3U, 384U,
                                get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/Relu");

        // Branch 2
        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0), true)).set_name("Mixed_6a/Branch_2/MaxPool_1a_3x3");

        // Concatenate
        graph << ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c)).set_name("Mixed_6a/concat");
    }

    void block_mixed_7a(const std::string &data_path, DataLayout weights_layout)
    {
        // Branch 0
        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_7a/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_0/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 384U,
                                get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/Relu");

        // Branch 1
        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 288U,
                                get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/Relu");

        // Branch 2
        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 0, 0))
            .set_name("Mixed_7a/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_7a/Branch_2/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(3U, 3U, 288U,
                                get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(1, 1, 1, 1))
            .set_name("Mixed_7a/Branch_2/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_7a/Branch_2/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_2/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(3U, 3U, 320U,
                                get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_weights.npy", weights_layout),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                PadStrideInfo(2, 2, 0, 0))
            .set_name("Mixed_7a/Branch_2/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.0010000000474974513f)
            .set_name("Mixed_7a/Branch_2/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Mixed_7a/Branch_2/Conv2d_1a_3x3/Relu");

        // Branch 3
        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true)).set_name("Mixed_7a/Branch_3/MaxPool_1a_3x3");

        // Concatenate
        graph << ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d)).set_name("Mixed_7a/concat");
    }

    void block35_repeat(const std::string &data_path, DataLayout weights_layout, unsigned int num_blocks)
    {
        for(unsigned int i = 0; i < num_blocks; ++i)
        {
            std::stringstream unit_path_ss;
            unit_path_ss << "Repeat_block35_" << (i + 1) << "_";
            std::stringstream unit_name_ss;
            unit_name_ss << "Repeat/block35_" << (i + 1) << "/";

            std::string unit_path = unit_path_ss.str();
            std::string unit_name = unit_name_ss.str();

            // Create left and write substreams
            SubStream i_l(graph);
            SubStream i_r(graph);

            // Branch 0
            SubStream i_la(i_l);
            i_la << ConvolutionLayer(1U, 1U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                 .set_name(unit_name + "Branch_0/Conv2d_1x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_0/Conv2d_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_0/Conv2d_1x1/Relu");

            // Branch 1
            SubStream i_lb(i_l);
            i_lb << ConvolutionLayer(1U, 1U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                 .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(3U, 3U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_3x3_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 1))
                 .set_name(unit_name + "Branch_1/Conv2d_0b_3x3/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0b_3x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0b_3x3/Relu");

            // Branch 2
            SubStream i_lc(i_l);
            i_lc << ConvolutionLayer(1U, 1U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                 .set_name(unit_name + "Branch_2/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_2/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_2/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(3U, 3U, 48U,
                                     get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 1))
                 .set_name(unit_name + "Branch_2/Conv2d_0b_3x3/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_2/Conv2d_0b_3x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_2/Conv2d_0b_3x3/Relu")
                 << ConvolutionLayer(3U, 3U, 64U,
                                     get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0c_3x3_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 1))
                 .set_name(unit_name + "Branch_2/Conv2d_0c_3x3/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_2/Conv2d_0c_3x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_2/Conv2d_0c_3x3/Relu");

            // Concatenate
            i_l << ConcatLayer(std::move(i_la), std::move(i_lb), std::move(i_lc)).set_name(unit_name + "concat")
                << ConvolutionLayer(1U, 1U, 320U,
                                    get_weights_accessor(data_path, unit_path + "Conv2d_1x1_weights.npy", weights_layout),
                                    get_weights_accessor(data_path, unit_path + "Conv2d_1x1_biases.npy", weights_layout),
                                    PadStrideInfo(1, 1, 0, 0))
                .set_name(unit_name + "Conv2d_1x1/convolution")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 0.17f, 0.f)).set_name(unit_name + "mul");

            graph << EltwiseLayer(std::move(i_l), std::move(i_r), EltwiseOperation::Add).set_name(unit_name + "add")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
        }
    }

    void block17_repeat(const std::string &data_path, DataLayout weights_layout, unsigned int num_blocks)
    {
        for(unsigned int i = 0; i < num_blocks; ++i)
        {
            std::stringstream unit_path_ss;
            unit_path_ss << "Repeat_1_block17_" << (i + 1) << "_";
            std::stringstream unit_name_ss;
            unit_name_ss << "Repeat_1/block17_" << (i + 1) << "/";

            std::string unit_path = unit_path_ss.str();
            std::string unit_name = unit_name_ss.str();

            // Create left and write substreams
            SubStream i_l(graph);
            SubStream i_r(graph);

            // Branch 0
            SubStream i_la(i_l);
            i_la << ConvolutionLayer(1U, 1U, 192U,
                                     get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                 .set_name(unit_name + "Branch_0/Conv2d_1x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_0/Conv2d_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_0/Conv2d_1x1/Relu");

            // Branch 1
            SubStream i_lb(i_l);
            i_lb << ConvolutionLayer(1U, 1U, 128U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                 .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(7U, 1U, 160U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x7_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 3, 0))
                 .set_name(unit_name + "Branch_1/Conv2d_0b_1x7/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0b_1x7/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0b_1x7/Relu")
                 << ConvolutionLayer(1U, 7U, 192U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_7x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 3))
                 .set_name(unit_name + "Branch_1/Conv2d_0c_7x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0c_7x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0c_7x1/Relu");

            // Concatenate
            i_l << ConcatLayer(std::move(i_la), std::move(i_lb)).set_name(unit_name + "concat")
                << ConvolutionLayer(1U, 1U, 1088U,
                                    get_weights_accessor(data_path, unit_path + "Conv2d_1x1_weights.npy", weights_layout),
                                    get_weights_accessor(data_path, unit_path + "Conv2d_1x1_biases.npy", weights_layout),
                                    PadStrideInfo(1, 1, 0, 0))
                .set_name(unit_name + "Conv2d_1x1/convolution")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 0.10f, 0.f)).set_name(unit_name + "mul");

            graph << EltwiseLayer(std::move(i_l), std::move(i_r), EltwiseOperation::Add).set_name(unit_name + "add")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
        }
    }

    void block8_repeat(const std::string &data_path, DataLayout weights_layout, unsigned int num_blocks, float scale, bool has_activation)
    {
        for(unsigned int i = 0; i < num_blocks; ++i)
        {
            std::stringstream unit_path_ss;
            std::stringstream unit_name_ss;
            if(num_blocks != 1)
            {
                unit_path_ss << "Repeat_2_block8_" << (i + 1) << "_";
                unit_name_ss << "Repeat_2/block8_" << (i + 1) << "/";
            }
            else
            {
                unit_path_ss << "Block8_";
                unit_name_ss << "Block8/";
            }

            std::string unit_path = unit_path_ss.str();
            std::string unit_name = unit_name_ss.str();

            // Create left and write substreams
            SubStream i_l(graph);
            SubStream i_r(graph);

            // Branch 0
            SubStream i_la(i_l);
            i_la << ConvolutionLayer(1U, 1U, 192U,
                                     get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                 .set_name(unit_name + "Branch_0/Conv2d_1x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_0/Conv2d_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_0/Conv2d_1x1/Relu");

            // Branch 1
            SubStream i_lb(i_l);
            i_lb << ConvolutionLayer(1U, 1U, 192U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                 .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(3U, 1U, 224U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x3_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 0))
                 .set_name(unit_name + "Branch_1/Conv2d_0b_1x3/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0b_1x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0b_1x3/Relu")
                 << ConvolutionLayer(1U, 3U, 256U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_3x1_weights.npy", weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 1))
                 .set_name(unit_name + "Branch_1/Conv2d_0c_3x1/convolution")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_variance.npy"),
                                            get_random_accessor(1.f, 1.f),
                                            get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(unit_name + "Branch_1/Conv2d_0c_3x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Branch_1/Conv2d_0c_3x1/Relu");

            // Concatenate
            i_l << ConcatLayer(std::move(i_la), std::move(i_lb)).set_name(unit_name + "concat")
                << ConvolutionLayer(1U, 1U, 2080U,
                                    get_weights_accessor(data_path, unit_path + "Conv2d_1x1_weights.npy", weights_layout),
                                    get_weights_accessor(data_path, unit_path + "Conv2d_1x1_biases.npy", weights_layout),
                                    PadStrideInfo(1, 1, 0, 0))
                .set_name(unit_name + "Conv2d_1x1/convolution");

            // Scale result
            if(scale != 1.f)
            {
                i_l << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, scale, 0.f)).set_name(unit_name + "mul");
            }

            // Residual add
            graph << EltwiseLayer(std::move(i_l), std::move(i_r), EltwiseOperation::Add).set_name(unit_name + "add");

            // Apply activation if needed
            if(has_activation)
            {
                graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
            }
        }
    }
};

/** Main program for Inception ResNet V2
 *
 * Model is based on:
 *      https://arxiv.org/abs/1602.07261
 *      "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
 *      Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
 *
 * Provenance: download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<InceptionResNetV2Example>(argc, argv);
}
