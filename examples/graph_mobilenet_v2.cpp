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

/** Example demonstrating how to implement MobileNetV2's network using the Compute Library's graph API */
class GraphMobilenetV2Example : public Example
{
public:
    GraphMobilenetV2Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "MobileNetV2")
    {
    }
    GraphMobilenetV2Example(const GraphMobilenetV2Example &) = delete;
    GraphMobilenetV2Example &operator=(const GraphMobilenetV2Example &) = delete;
    GraphMobilenetV2Example(GraphMobilenetV2Example &&)                 = default; // NOLINT
    GraphMobilenetV2Example &operator=(GraphMobilenetV2Example &&) = default;      // NOLINT
    ~GraphMobilenetV2Example() override                            = default;

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

        // Create model path
        std::string model_path = "/cnn_data/mobilenet_v2_1.0_224_model/";

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        // Create graph
        graph << common_params.target
              << DepthwiseConvolutionMethod::Optimized3x3 // FIXME(COMPMID-1073): Add heuristics to automatically call the optimized 3x3 method
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false))
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "Conv_weights.npy", DataLayout::NCHW),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL))
              .set_name("Conv")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv_BatchNorm_moving_variance.npy"),
                                         get_weights_accessor(data_path, "Conv_BatchNorm_gamma.npy"),
                                         get_weights_accessor(data_path, "Conv_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
              .set_name("Conv/Relu6");

        get_expanded_conv(data_path, "expanded_conv", 32U, 16U, PadStrideInfo(1, 1, 1, 1));
        get_expanded_conv(data_path, "expanded_conv_1", 16U, 24U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), true);
        get_expanded_conv(data_path, "expanded_conv_2", 24U, 24U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_3", 24U, 32U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), true);
        get_expanded_conv(data_path, "expanded_conv_4", 32U, 32U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_5", 32U, 32U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_6", 32U, 64U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), true);
        get_expanded_conv(data_path, "expanded_conv_7", 64U, 64U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_8", 64U, 64U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_9", 64U, 64U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_10", 64U, 96U, PadStrideInfo(1, 1, 1, 1), true);
        get_expanded_conv(data_path, "expanded_conv_11", 96U, 96U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_12", 96U, 96U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_13", 96U, 160U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), true);
        get_expanded_conv(data_path, "expanded_conv_14", 160U, 160U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_15", 160U, 160U, PadStrideInfo(1, 1, 1, 1), true, true);
        get_expanded_conv(data_path, "expanded_conv_16", 160U, 320U, PadStrideInfo(1, 1, 1, 1), true);

        graph << ConvolutionLayer(1U, 1U, 1280U,
                                  get_weights_accessor(data_path, "Conv_1_weights.npy", DataLayout::NCHW),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Conv_1")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "Conv_1_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "Conv_1_BatchNorm_moving_variance.npy"),
                                         get_weights_accessor(data_path, "Conv_1_BatchNorm_gamma.npy"),
                                         get_weights_accessor(data_path, "Conv_1_BatchNorm_beta.npy"),
                                         0.0010000000474974513f)
              .set_name("Conv_1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
              .set_name("Conv_1/Relu6")
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("Logits/AvgPool")
              << ConvolutionLayer(1U, 1U, 1001U,
                                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy", DataLayout::NCHW),
                                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_biases.npy"),
                                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Logits/Conv2d_1c_1x1")
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
        // Run graph
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    void get_expanded_conv(const std::string &data_path, std::string &&param_path,
                           unsigned int input_channels, unsigned int output_channels,
                           PadStrideInfo dwc_pad_stride_info,
                           bool has_expand = false, bool is_residual = false, unsigned int expansion_size = 6)
    {
        std::string total_path = param_path + "_";
        SubStream   left(graph);

        // Add expand node
        if(has_expand)
        {
            left << ConvolutionLayer(1U, 1U, input_channels * expansion_size,
                                     get_weights_accessor(data_path, total_path + "expand_weights.npy", DataLayout::NCHW),
                                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
                 .set_name(param_path + "/expand/Conv2D")
                 << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "expand_BatchNorm_moving_mean.npy"),
                                            get_weights_accessor(data_path, total_path + "expand_BatchNorm_moving_variance.npy"),
                                            get_weights_accessor(data_path, total_path + "expand_BatchNorm_gamma.npy"),
                                            get_weights_accessor(data_path, total_path + "expand_BatchNorm_beta.npy"),
                                            0.0010000000474974513f)
                 .set_name(param_path + "/expand/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
                 .set_name(param_path + "/expand/Relu6");
        }

        // Add depthwise node
        left << DepthwiseConvolutionLayer(3U, 3U,
                                          get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy", DataLayout::NCHW),
                                          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                          dwc_pad_stride_info)
             .set_name(param_path + "/depthwise/depthwise")
             << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_mean.npy"),
                                        get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_variance.npy"),
                                        get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_gamma.npy"),
                                        get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_beta.npy"),
                                        0.0010000000474974513f)
             .set_name(param_path + "/depthwise/BatchNorm")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
             .set_name(param_path + "/depthwise/Relu6");

        // Add project node
        left << ConvolutionLayer(1U, 1U, output_channels,
                                 get_weights_accessor(data_path, total_path + "project_weights.npy", DataLayout::NCHW),
                                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
             .set_name(param_path + "/project/Conv2D")
             << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "project_BatchNorm_moving_mean.npy"),
                                        get_weights_accessor(data_path, total_path + "project_BatchNorm_moving_variance.npy"),
                                        get_weights_accessor(data_path, total_path + "project_BatchNorm_gamma.npy"),
                                        get_weights_accessor(data_path, total_path + "project_BatchNorm_beta.npy"),
                                        0.0010000000474974513)
             .set_name(param_path + "/project/BatchNorm");

        if(is_residual)
        {
            // Add residual node
            SubStream right(graph);
            graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(param_path + "/add");
        }
        else
        {
            graph.forward_tail(left.tail_node());
        }
    }
};

/** Main program for MobileNetV2
 *
 * Model is based on:
 *      https://arxiv.org/abs/1801.04381
 *      "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
 *      Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphMobilenetV2Example>(argc, argv);
}
