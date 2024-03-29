/*
 * Copyright (c) 2018-2021 Arm Limited.
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

const float batch_norm_epsilon = 0.0010000000474974513f;

/** Example demonstrating how to implement Inception ResNet V1 network using the Compute Library's graph API */
class InceptionResNetV1Example final : public Example
{
public:
    InceptionResNetV1Example()
        : cmd_parser(),
          common_opts(cmd_parser),
          common_params(),
          model_input_width(nullptr),
          model_input_height(nullptr),
          graph(0, "InceptionResNetV1")
    {
        model_input_width  = cmd_parser.add_option<SimpleOption<unsigned int>>("image-width", 512);
        model_input_height = cmd_parser.add_option<SimpleOption<unsigned int>>("image-height", 512);

        // Add model id option
        model_input_width->set_help("Input image width.");
        model_input_height->set_help("Input image height.");
    }
    InceptionResNetV1Example(const InceptionResNetV1Example &)            = delete;
    InceptionResNetV1Example &operator=(const InceptionResNetV1Example &) = delete;
    ~InceptionResNetV1Example() override                                  = default;
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if (common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }
        // Get input image width and height
        const unsigned int image_width  = model_input_width->value();
        const unsigned int image_height = model_input_height->value();

        // Set default layout if needed
        if (!common_opts.data_layout->is_set() && common_params.target == Target::NEON)
        {
            common_params.data_layout = DataLayout::NCHW;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type),
                                "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;
        std::cout << "Image width: " << image_width << std::endl;
        std::cout << "Image height: " << image_height << std::endl;

        // Create model path
        std::string data_path  = common_params.data_path;
        std::string model_path = "/cnn_data/inception_resnet_v1_model/";
        if (!data_path.empty())
        {
            data_path += model_path;
        }

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<TFPreproccessor>(0.f, 1.f);

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape     = permute_shape(
                TensorShape(image_width, image_height, 3U, common_params.batches), DataLayout::NCHW, operation_layout);
        TensorDescriptor input_descriptor =
            TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false))
              // Conv2d_1a_3x3
              << ConvolutionLayer(
                     3U, 3U, 32U, get_weights_accessor(data_path, "Conv2d_1a_3x3_weights.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                     .set_name("Conv2d_1a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                         batch_norm_epsilon)
                     .set_name("Conv2d_1a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("Conv2d_1a_3x3/Relu")
              // Conv2d_2a_3x3
              << ConvolutionLayer(
                     3U, 3U, 32U, get_weights_accessor(data_path, "Conv2d_2a_3x3_weights.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                     .set_name("Conv2d_2a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_2a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_2a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_2a_3x3_BatchNorm_beta.npy"),
                                         batch_norm_epsilon)
                     .set_name("Conv2d_2a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("Conv2d_2a_3x3/Relu")
              // Conv2d_2b_3x3
              << ConvolutionLayer(
                     3U, 3U, 64U, get_weights_accessor(data_path, "Conv2d_2b_3x3_weights.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
                     .set_name("Conv2d_2b_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_2b_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_2b_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_2b_3x3_BatchNorm_beta.npy"),
                                         batch_norm_epsilon)
                     .set_name("Conv2d_2b_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("Conv2d_2b_3x3/Relu")
              // MaxPool_3a_3x3
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout,
                                               PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true))
                     .set_name("MaxPool_3a_3x3/MaxPool")
              // Conv2d_3b_1x1
              << ConvolutionLayer(
                     1U, 1U, 80U, get_weights_accessor(data_path, "Conv2d_3b_1x1_weights.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                     .set_name("Conv2d_3b_1x1/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_3b_1x1_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_3b_1x1_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_3b_1x1_BatchNorm_beta.npy"),
                                         batch_norm_epsilon)
                     .set_name("Conv2d_3b_1x1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("Conv2d_3b_1x1/Relu")
              // Conv2d_4a_3x3
              << ConvolutionLayer(
                     3U, 3U, 192U, get_weights_accessor(data_path, "Conv2d_4a_3x3_weights.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                     .set_name("Conv2d_4a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_4a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_4a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_4a_3x3_BatchNorm_beta.npy"),
                                         batch_norm_epsilon)
                     .set_name("Conv2d_4a_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("Conv2d_4a_3x3/Relu")
              // Conv2d_4b_3x3
              << ConvolutionLayer(
                     3U, 3U, 256U, get_weights_accessor(data_path, "Conv2d_4b_3x3_weights.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                     .set_name("Conv2d_4a_3x3/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv2d_4b_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv2d_4b_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "Conv2d_4b_3x3_BatchNorm_beta.npy"),
                                         batch_norm_epsilon)
                     .set_name("Conv2d_4b_3x3/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                     .set_name("Conv2d_4b_3x3/Relu");

        // 5 x Inception-resnet-A
        block35_repeat(data_path, weights_layout, 5);
        // Reduction-A
        reduction_a(data_path, weights_layout);
        // 10 x Inception-Resnet-B
        block17_repeat(data_path, weights_layout, 10);
        // Reduction-B
        reduction_b(data_path, weights_layout);
        // 5 x Inception-resnet-C
        block8_repeat(data_path, weights_layout, 5, 0.2f, true);

        block8_repeat(data_path, weights_layout, 1, 1.f, false);

        // Logits tail
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, operation_layout)).set_name("Logits/AvgPool_1a_8x8")
              << FlattenLayer().set_name("Logits/Flatten")
              << FullyConnectedLayer(128U, get_weights_accessor(data_path, "Logits_Logits_weights.npy", weights_layout),
                                     get_weights_accessor(data_path, "Logits_Logits_biases.npy"))
                     .set_name("Logits/Logits")
              << OutputLayer(std::make_unique<DummyAccessor>(0));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        graph.finalize(common_params.target, config);

        return true;
    }

    void do_run() override
    {
        graph.run();
    }

private:
    CommandLineParser           cmd_parser;
    CommonGraphOptions          common_opts;
    CommonGraphParams           common_params;
    SimpleOption<unsigned int> *model_input_width{nullptr};
    SimpleOption<unsigned int> *model_input_height{nullptr};
    Stream                      graph;

private:
    void block35_repeat(const std::string &data_path, DataLayout weights_layout, unsigned int num_blocks)
    {
        for (unsigned int i = 0; i < num_blocks; ++i)
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
            i_la << ConvolutionLayer(
                        1U, 1U, 32U,
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_weights.npy", weights_layout),
                        std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/Relu");

            // Branch 1
            SubStream i_lb(i_l);
            i_lb << ConvolutionLayer(1U, 1U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(3U, 3U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_3x3_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 1))
                        .set_name(unit_name + "Branch_1/Conv2d_0b_3x3/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0b_3x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0b_3x3/Relu");

            // Branch 2
            SubStream i_lc(i_l);
            i_lc << ConvolutionLayer(1U, 1U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0a_1x1_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                        .set_name(unit_name + "Branch_2/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_2/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_2/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(3U, 3U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0b_3x3_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 1))
                        .set_name(unit_name + "Branch_2/Conv2d_0b_3x3/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_2/Conv2d_0b_3x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_2/Conv2d_0b_3x3/Relu")
                 << ConvolutionLayer(3U, 3U, 32U,
                                     get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0c_3x3_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 1))
                        .set_name(unit_name + "Branch_2/Conv2d_0c_3x3/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_2/Conv2d_0c_3x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_2/Conv2d_0c_3x3/Relu");

            // Concatenate
            i_l << ConcatLayer(std::move(i_la), std::move(i_lb), std::move(i_lc)).set_name(unit_name + "concat")
                << ConvolutionLayer(
                       1U, 1U, 256U,
                       get_weights_accessor(data_path, unit_path + "Conv2d_1x1_weights.npy", weights_layout),
                       get_weights_accessor(data_path, unit_path + "Conv2d_1x1_biases.npy", weights_layout),
                       PadStrideInfo(1, 1, 0, 0))
                       .set_name(unit_name + "Conv2d_1x1/convolution")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 0.17f, 0.f))
                       .set_name(unit_name + "mul");

            graph << EltwiseLayer(std::move(i_l), std::move(i_r), EltwiseOperation::Add).set_name(unit_name + "add")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                         .set_name(unit_name + "Relu");
        }
    }

    void block17_repeat(const std::string &data_path, DataLayout weights_layout, unsigned int num_blocks)
    {
        for (unsigned int i = 0; i < num_blocks; ++i)
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
            i_la << ConvolutionLayer(
                        1U, 1U, 128U,
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_weights.npy", weights_layout),
                        std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/Relu");

            // Branch 1
            SubStream i_lb(i_l);
            i_lb << ConvolutionLayer(1U, 1U, 128U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(7U, 1U, 128U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x7_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 3, 0))
                        .set_name(unit_name + "Branch_1/Conv2d_0b_1x7/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0b_1x7/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0b_1x7/Relu")
                 << ConvolutionLayer(1U, 7U, 128U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_7x1_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 3))
                        .set_name(unit_name + "Branch_1/Conv2d_0c_7x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0c_7x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0c_7x1/Relu");

            // Concatenate
            i_l << ConcatLayer(std::move(i_la), std::move(i_lb)).set_name(unit_name + "concat")
                << ConvolutionLayer(
                       1U, 1U, 896U,
                       get_weights_accessor(data_path, unit_path + "Conv2d_1x1_weights.npy", weights_layout),
                       get_weights_accessor(data_path, unit_path + "Conv2d_1x1_biases.npy", weights_layout),
                       PadStrideInfo(1, 1, 0, 0))
                       .set_name(unit_name + "Conv2d_1x1/convolution")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 0.10f, 0.f))
                       .set_name(unit_name + "mul");

            graph << EltwiseLayer(std::move(i_l), std::move(i_r), EltwiseOperation::Add).set_name(unit_name + "add")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                         .set_name(unit_name + "Relu");
        }
    }

    void block8_repeat(const std::string &data_path,
                       DataLayout         weights_layout,
                       unsigned int       num_blocks,
                       float              scale,
                       bool               has_activation)
    {
        for (unsigned int i = 0; i < num_blocks; ++i)
        {
            std::stringstream unit_path_ss;
            std::stringstream unit_name_ss;
            if (num_blocks != 1)
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
            i_la << ConvolutionLayer(
                        1U, 1U, 192U,
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_weights.npy", weights_layout),
                        std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_0_Conv2d_1x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_0_Conv2d_1x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_0/Conv2d_1x1/Relu");

            // Branch 1
            SubStream i_lb(i_l);
            i_lb << ConvolutionLayer(1U, 1U, 192U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 0))
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0a_1x1/Relu")
                 << ConvolutionLayer(3U, 1U, 192U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x3_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 1, 0))
                        .set_name(unit_name + "Branch_1/Conv2d_0b_1x3/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0b_1x3/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0b_1x3/Relu")
                 << ConvolutionLayer(1U, 3U, 192U,
                                     get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_3x1_weights.npy",
                                                          weights_layout),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                     PadStrideInfo(1, 1, 0, 1))
                        .set_name(unit_name + "Branch_1/Conv2d_0c_3x1/convolution")
                 << BatchNormalizationLayer(
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_mean.npy"),
                        get_weights_accessor(data_path,
                                             unit_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_variance.npy"),
                        get_random_accessor(1.f, 1.f),
                        get_weights_accessor(data_path, unit_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_beta.npy"),
                        batch_norm_epsilon)
                        .set_name(unit_name + "Branch_1/Conv2d_0c_3x1/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                        .set_name(unit_name + "Branch_1/Conv2d_0c_3x1/Relu");

            // Concatenate
            i_l << ConcatLayer(std::move(i_la), std::move(i_lb)).set_name(unit_name + "concat")
                << ConvolutionLayer(
                       1U, 1U, 1792U,
                       get_weights_accessor(data_path, unit_path + "Conv2d_1x1_weights.npy", weights_layout),
                       get_weights_accessor(data_path, unit_path + "Conv2d_1x1_biases.npy", weights_layout),
                       PadStrideInfo(1, 1, 0, 0))
                       .set_name(unit_name + "Conv2d_1x1/convolution");

            // Scale result
            if (scale != 1.f)
            {
                i_l << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, scale, 0.f))
                           .set_name(unit_name + "mul");
            }

            // Residual add
            graph << EltwiseLayer(std::move(i_l), std::move(i_r), EltwiseOperation::Add).set_name(unit_name + "add");

            // Apply activation if needed
            if (has_activation)
            {
                graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                             .set_name(unit_name + "Relu");
            }
        }
    }

    void reduction_a(const std::string &data_path, DataLayout weights_layout)
    {
        // Branch 0
        SubStream i_a(graph);
        i_a << ConvolutionLayer(
                   3U, 3U, 384U,
                   get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_6a/Branch_0/Conv2d_1a_3x3/Relu");

        // Branch 1
        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   1U, 1U, 192U,
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_6a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, 192U,
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
                   .set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_6a/Branch_1/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(
                   3U, 3U, 256U,
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_6a/Branch_1/Conv2d_1a_3x3/Relu");

        // Branch 2
        SubStream i_c(graph);
        i_c << PoolingLayer(
                   PoolingLayerInfo(PoolingType::MAX, 3, common_params.data_layout, PadStrideInfo(2, 2, 0, 0), true))
                   .set_name("Mixed_6a/Branch_2/MaxPool_1a_3x3");

        // Concatenate
        graph << ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c)).set_name("Mixed_6a/concat");
    }

    void reduction_b(const std::string &data_path, DataLayout weights_layout)
    {
        // Branch 0
        SubStream i_a(graph);
        i_a << ConvolutionLayer(
                   1U, 1U, 256U,
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name("Mixed_7a/Branch_0/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_7a/Branch_0/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, 384U,
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_7a/Branch_0/Conv2d_1a_3x3/Relu");

        // Branch 1
        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   1U, 1U, 256U,
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_7a/Branch_1/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, 256U,
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_7a/Branch_1/Conv2d_1a_3x3/Relu");

        // Branch 2
        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                   1U, 1U, 256U,
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                   .set_name("Mixed_7a/Branch_2/Conv2d_0a_1x1/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_7a/Branch_2/Conv2d_0a_1x1/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_7a/Branch_2/Conv2d_0a_1x1/Relu")
            << ConvolutionLayer(
                   3U, 3U, 256U,
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
                   .set_name("Mixed_7a/Branch_2/Conv2d_0b_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_7a/Branch_2/Conv2d_0b_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_7a/Branch_2/Conv2d_0b_3x3/Relu")
            << ConvolutionLayer(
                   3U, 3U, 256U,
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_weights.npy", weights_layout),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
                   .set_name("Mixed_7a/Branch_2/Conv2d_1a_3x3/convolution")
            << BatchNormalizationLayer(
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, "Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                   batch_norm_epsilon)
                   .set_name("Mixed_7a/Branch_2/Conv2d_1a_3x3/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
                   .set_name("Mixed_7a/Branch_2/Conv2d_1a_3x3/Relu");

        // Branch 3
        SubStream i_d(graph);
        i_d << PoolingLayer(
                   PoolingLayerInfo(PoolingType::MAX, 3, common_params.data_layout, PadStrideInfo(2, 2, 0, 0), true))
                   .set_name("Mixed_7a/Branch_3/MaxPool_1a_3x3");

        // Concatenate
        graph
            << ConcatLayer(std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d)).set_name("Mixed_7a/concat");
    }
};

/** Main program for Inception ResNet V1
 *
 * Model is based on:
 *      https://arxiv.org/abs/1602.07261
 *      "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
 *      Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<InceptionResNetV1Example>(argc, argv);
}
