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

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement MobileNet's network using the Compute Library's graph API */
class GraphMobilenetExample : public Example
{
public:
    GraphMobilenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "MobileNetV1")
    {
        // Add model id option
        model_id_opt = cmd_parser.add_option<SimpleOption<int>>("model-id", 0);
        model_id_opt->set_help("Mobilenet model id (0: 1.0_224, else: 0.75_160");
    }
    GraphMobilenetExample(const GraphMobilenetExample &) = delete;
    GraphMobilenetExample &operator=(const GraphMobilenetExample &) = delete;
    GraphMobilenetExample(GraphMobilenetExample &&)                 = default; // NOLINT
    GraphMobilenetExample &operator=(GraphMobilenetExample &&) = default;      // NOLINT
    ~GraphMobilenetExample() override                          = default;
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

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get model parameters
        int model_id = model_id_opt->value();

        // Create input descriptor
        unsigned int spatial_size = (model_id == 0 || common_params.data_type == DataType::QASYMM8) ? 224 : 160;

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(spatial_size, spatial_size, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set graph hints
        graph << common_params.target
              << DepthwiseConvolutionMethod::Optimized3x3 // TODO(COMPMID-1073): Add heuristics to automatically call the optimized 3x3 method
              << common_params.fast_math_hint;

        // Create core graph
        if(arm_compute::is_data_type_float(common_params.data_type))
        {
            create_graph_float(input_descriptor, model_id);
        }
        else
        {
            create_graph_qasymm(input_descriptor);
        }

        // Create common tail
        graph << ReshapeLayer(TensorShape(1001U)).set_name("Reshape")
              << SoftmaxLayer().set_name("Softmax")
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
    SimpleOption<int> *model_id_opt{ nullptr };
    CommonGraphParams  common_params;
    Stream             graph;

    void create_graph_float(TensorDescriptor &input_descriptor, int model_id)
    {
        float       depth_scale = (model_id == 0) ? 1.f : 0.75;
        std::string model_path  = (model_id == 0) ? "/cnn_data/mobilenet_v1_1_224_model/" : "/cnn_data/mobilenet_v1_075_160_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        graph << InputLayer(input_descriptor,
                            get_input_accessor(common_params, std::move(preprocessor), false))
              << ConvolutionLayer(
                  3U, 3U, 32U * depth_scale,
                  get_weights_accessor(data_path, "Conv2d_0_weights.npy", DataLayout::NCHW),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
              .set_name("Conv2d_0")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_BatchNorm_beta.npy"),
                  0.001f)
              .set_name("Conv2d_0/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name("Conv2d_0/Relu6");
        graph << get_dwsc_node_float(data_path, "Conv2d_1", 64 * depth_scale, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_2", 128 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_3", 128 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_4", 256 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_5", 256 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_6", 512 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_7", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_8", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_9", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_10", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_11", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_12", 1024 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_13", 1024 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("Logits/AvgPool_1a")
              << ConvolutionLayer(
                  1U, 1U, 1001U,
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy", DataLayout::NCHW),
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_biases.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("Logits/Conv2d_1c_1x1");
    }

    void create_graph_qasymm(TensorDescriptor &input_descriptor)
    {
        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += "/cnn_data/mobilenet_qasymm8_model/";
        }

        // Quantization info taken from the AndroidNN QASYMM8 MobileNet example
        const QuantizationInfo in_quant_info = QuantizationInfo(0.0078125f, 128);

        const std::vector<QuantizationInfo> conv_weights_quant_info =
        {
            QuantizationInfo(0.02182667888700962f, 151), // conv0
            QuantizationInfo(0.004986600950360298f, 74)  // conv14
        };
        const std::vector<QuantizationInfo> conv_out_quant_info =
        {
            QuantizationInfo(0.023528477177023888f, 0), // conv0
            QuantizationInfo(0.16609922051429749f, 66)  // conv14
        };

        const std::vector<QuantizationInfo> depth_weights_quant_info =
        {
            QuantizationInfo(0.29219913482666016f, 110),  // dwsc1
            QuantizationInfo(0.40277284383773804f, 130),  // dwsc2
            QuantizationInfo(0.06053730100393295f, 160),  // dwsc3
            QuantizationInfo(0.01675807684659958f, 123),  // dwsc4
            QuantizationInfo(0.04105526953935623f, 129),  // dwsc5
            QuantizationInfo(0.013460792601108551f, 122), // dwsc6
            QuantizationInfo(0.036934755742549896f, 132), // dwsc7
            QuantizationInfo(0.042609862983226776f, 94),  // dwsc8
            QuantizationInfo(0.028358859941363335f, 127), // dwsc9
            QuantizationInfo(0.024329448118805885f, 134), // dwsc10
            QuantizationInfo(0.019366811960935593f, 106), // dwsc11
            QuantizationInfo(0.007835594937205315f, 126), // dwsc12
            QuantizationInfo(0.12616927921772003f, 211)   // dwsc13
        };

        const std::vector<QuantizationInfo> point_weights_quant_info =
        {
            QuantizationInfo(0.030420949682593346f, 121), // dwsc1
            QuantizationInfo(0.015148180536925793f, 104), // dwsc2
            QuantizationInfo(0.013755458407104015f, 94),  // dwsc3
            QuantizationInfo(0.007601846940815449f, 151), // dwsc4
            QuantizationInfo(0.006431614048779011f, 122), // dwsc5
            QuantizationInfo(0.00917122047394514f, 109),  // dwsc6
            QuantizationInfo(0.005300046876072884f, 140), // dwsc7
            QuantizationInfo(0.0049632852897048f, 127),   // dwsc8
            QuantizationInfo(0.007770895957946777f, 89),  // dwsc9
            QuantizationInfo(0.009658650495111942f, 99),  // dwsc10
            QuantizationInfo(0.005446993745863438f, 153), // dwsc11
            QuantizationInfo(0.00817922968417406f, 130),  // dwsc12
            QuantizationInfo(0.018048152327537537f, 95)   // dwsc13
        };

        graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info),
                            get_weights_accessor(data_path, common_params.image))
              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "Conv2d_0_weights.npy"),
                  get_weights_accessor(data_path, "Conv2d_0_bias.npy"),
                  PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
                  1, conv_weights_quant_info.at(0), conv_out_quant_info.at(0))
              .set_name("Conv2d_0")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name("Conv2d_0/Relu6");
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_1", 64U, PadStrideInfo(1U, 1U, 1U, 1U), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(0), point_weights_quant_info.at(0));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_2", 128U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(1),
                                      point_weights_quant_info.at(1));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_3", 128U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(2),
                                      point_weights_quant_info.at(2));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_4", 256U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(3),
                                      point_weights_quant_info.at(3));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_5", 256U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(4),
                                      point_weights_quant_info.at(4));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_6", 512U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(5),
                                      point_weights_quant_info.at(5));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_7", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(6),
                                      point_weights_quant_info.at(6));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_8", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(7),
                                      point_weights_quant_info.at(7));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_9", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(8),
                                      point_weights_quant_info.at(8));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_10", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(9),
                                      point_weights_quant_info.at(9));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_11", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(10),
                                      point_weights_quant_info.at(10));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_12", 1024U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(11),
                                      point_weights_quant_info.at(11));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_13", 1024U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(12),
                                      point_weights_quant_info.at(12))
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("Logits/AvgPool_1a")
              << ConvolutionLayer(
                  1U, 1U, 1001U,
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy"),
                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_bias.npy"),
                  PadStrideInfo(1U, 1U, 0U, 0U), 1, conv_weights_quant_info.at(1), conv_out_quant_info.at(1))
              .set_name("Logits/Conv2d_1c_1x1");
    }

    ConcatLayer get_dwsc_node_float(const std::string &data_path, std::string &&param_path,
                                    unsigned int  conv_filt,
                                    PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
    {
        std::string total_path = param_path + "_";
        SubStream   sg(graph);
        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy", DataLayout::NCHW),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               dwc_pad_stride_info)
           .set_name(total_path + "depthwise/depthwise")
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_gamma.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_beta.npy"),
               0.001f)
           .set_name(total_path + "depthwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "depthwise/Relu6")
           << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "pointwise_weights.npy", DataLayout::NCHW),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               conv_pad_stride_info)
           .set_name(total_path + "pointwise/Conv2D")
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_gamma.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_beta.npy"),
               0.001f)
           .set_name(total_path + "pointwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "pointwise/Relu6");

        return ConcatLayer(std::move(sg));
    }

    ConcatLayer get_dwsc_node_qasymm(const std::string &data_path, std::string &&param_path,
                                     const unsigned int conv_filt,
                                     PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info,
                                     QuantizationInfo depth_weights_quant_info, QuantizationInfo point_weights_quant_info)
    {
        std::string total_path = param_path + "_";
        SubStream   sg(graph);

        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "depthwise_weights.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_bias.npy"),
               dwc_pad_stride_info, 1, std::move(depth_weights_quant_info))
           .set_name(total_path + "depthwise/depthwise")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name(total_path + "depthwise/Relu6")
           << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "pointwise_weights.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_bias.npy"),
               conv_pad_stride_info, 1, std::move(point_weights_quant_info))
           .set_name(total_path + "pointwise/Conv2D")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name(total_path + "pointwise/Relu6");

        return ConcatLayer(std::move(sg));
    }
};

/** Main program for MobileNetV1
 *
 * Model is based on:
 *      https://arxiv.org/abs/1704.04861
 *      "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
 *      Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
 *
 * Provenance: download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
 *             download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160.tgz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphMobilenetExample>(argc, argv);
}
