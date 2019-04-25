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

        // Print parameter values
        std::cout << common_params << std::endl;

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set graph hints
        graph << common_params.target
              << DepthwiseConvolutionMethod::Optimized3x3 // TODO(COMPMID-1073): Add heuristics to automatically call the optimized 3x3 method
              << common_params.fast_math_hint;

        // Create core graph
        if(arm_compute::is_data_type_float(common_params.data_type))
        {
            create_graph_float(input_descriptor);
        }
        else
        {
            create_graph_qasymm8(input_descriptor);
        }
        // Create common tail
        graph << ReshapeLayer(TensorShape(1001U)).set_name("Predictions/Reshape")
              << SoftmaxLayer().set_name("Predictions/Softmax")
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

private:
    enum class IsResidual
    {
        Yes,
        No
    };

    enum class HasExpand
    {
        Yes,
        No
    };

private:
    void create_graph_float(TensorDescriptor &input_descriptor)
    {
        // Create model path
        const std::string model_path = "/cnn_data/mobilenet_v2_1.0_224_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        graph << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false))
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

        get_expanded_conv_float(data_path, "expanded_conv", 32U, 16U, PadStrideInfo(1, 1, 1, 1));
        get_expanded_conv_float(data_path, "expanded_conv_1", 16U, 24U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), HasExpand::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_2", 24U, 24U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_3", 24U, 32U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), HasExpand::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_4", 32U, 32U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_5", 32U, 32U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_6", 32U, 64U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), HasExpand::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_7", 64U, 64U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_8", 64U, 64U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_9", 64U, 64U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_10", 64U, 96U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_11", 96U, 96U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_12", 96U, 96U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_13", 96U, 160U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), HasExpand::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_14", 160U, 160U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_15", 160U, 160U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes, IsResidual::Yes);
        get_expanded_conv_float(data_path, "expanded_conv_16", 160U, 320U, PadStrideInfo(1, 1, 1, 1), HasExpand::Yes);

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
              .set_name("Logits/Conv2d_1c_1x1");
    }

    void get_expanded_conv_float(const std::string &data_path, std::string &&param_path,
                                 unsigned int input_channels, unsigned int output_channels,
                                 PadStrideInfo dwc_pad_stride_info,
                                 HasExpand has_expand = HasExpand::No, IsResidual is_residual = IsResidual::No,
                                 unsigned int expansion_size = 6)
    {
        std::string total_path = param_path + "_";
        SubStream   left(graph);

        // Add expand node
        if(has_expand == HasExpand::Yes)
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

        if(is_residual == IsResidual::Yes)
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

    void create_graph_qasymm8(TensorDescriptor &input_descriptor)
    {
        // Create model path
        const std::string model_path = "/cnn_data/mobilenet_v2_1.0_224_quantized_model/";

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        const QuantizationInfo in_quant_info  = QuantizationInfo(0.0078125f, 128);
        const QuantizationInfo mid_quant_info = QuantizationInfo(0.023528477177023888f, 128);

        const std::vector<QuantizationInfo> conv_weights_quant_info =
        {
            QuantizationInfo(0.03396892547607422f, 122),  // Conv
            QuantizationInfo(0.005167067516595125f, 125), // Conv1
            QuantizationInfo(0.0016910821432247758f, 113) // Conv2d_1c_1x1
        };

        // Pointwise expand convolution quantization info
        const std::vector<QuantizationInfo> pwc_q =
        {
            QuantizationInfo(0.254282623529f, 129),        // expand_0 (Dummy)
            QuantizationInfo(0.009758507832884789f, 127),  // expand_1
            QuantizationInfo(0.0036556976847350597f, 144), // expand_2
            QuantizationInfo(0.0029988749884068966f, 104), // expand_3
            QuantizationInfo(0.0019244228024035692f, 128), // expand_4
            QuantizationInfo(0.0013649158645421267f, 135), // expand_5
            QuantizationInfo(0.0019170437008142471f, 127), // expand_6
            QuantizationInfo(0.0015538912266492844f, 125), // expand_7
            QuantizationInfo(0.0014702979242429137f, 134), // expand_8
            QuantizationInfo(0.0013733493397012353f, 127), // expand_9
            QuantizationInfo(0.0016282502328976989f, 131), // expand_10
            QuantizationInfo(0.0016309921629726887f, 134), // expand_11
            QuantizationInfo(0.0018258779309689999f, 138), // expand_12
            QuantizationInfo(0.0013828007504343987f, 123), // expand_13
            QuantizationInfo(0.0020222084131091833f, 135), // expand_14
            QuantizationInfo(0.04281935095787048f, 102),   // expand_15
            QuantizationInfo(0.002046825597062707f, 135)   // expand_16
        };
        // Depthwise expand convolution quantization info
        const std::vector<QuantizationInfo> dwc_q =
        {
            QuantizationInfo(0.3436955213546753f, 165),   // expand_0
            QuantizationInfo(0.020969120785593987f, 109), // expand_1
            QuantizationInfo(0.16981913149356842f, 52),   // expand_2
            QuantizationInfo(0.017202870920300484f, 143), // expand_3
            QuantizationInfo(0.06525065749883652f, 118),  // expand_4
            QuantizationInfo(0.07909784466028214f, 95),   // expand_5
            QuantizationInfo(0.010087885893881321f, 127), // expand_6
            QuantizationInfo(0.06092711538076401f, 110),  // expand_7
            QuantizationInfo(0.052407849580049515f, 133), // expand_8
            QuantizationInfo(0.04077887907624245f, 155),  // expand_9
            QuantizationInfo(0.031107846647500992f, 143), // expand_10
            QuantizationInfo(0.07080810517072678f, 66),   // expand_11
            QuantizationInfo(0.07448793947696686f, 159),  // expand_12
            QuantizationInfo(0.01525793131440878f, 92),   // expand_13
            QuantizationInfo(0.04166752099990845f, 147),  // expand_14
            QuantizationInfo(0.04281935095787048f, 102),  // expand_15
            QuantizationInfo(0.16456253826618195, 201)    // expand_16
        };
        // Project convolution quantization info
        const std::vector<QuantizationInfo> prwc_q =
        {
            QuantizationInfo(0.03737175464630127f, 140),  // expand_0
            QuantizationInfo(0.0225360207259655f, 156),   // expand_1
            QuantizationInfo(0.02740888111293316f, 122),  // expand_2
            QuantizationInfo(0.016844693571329117f, 111), // expand_3
            QuantizationInfo(0.019062912091612816f, 146), // expand_4
            QuantizationInfo(0.018293123692274094f, 128), // expand_5
            QuantizationInfo(0.014601286500692368f, 147), // expand_6
            QuantizationInfo(0.016782939434051514f, 124), // expand_7
            QuantizationInfo(0.012898261658847332f, 125), // expand_8
            QuantizationInfo(0.019561484456062317f, 144), // expand_9
            QuantizationInfo(0.007436311338096857f, 129), // expand_10
            QuantizationInfo(0.00838223285973072f, 136),  // expand_11
            QuantizationInfo(0.023982593789696693f, 154), // expand_12
            QuantizationInfo(0.009447949007153511f, 140), // expand_13
            QuantizationInfo(0.00789870135486126f, 139),  // expand_14
            QuantizationInfo(0.03697410225868225f, 131),  // expand_15
            QuantizationInfo(0.008009289391338825f, 111)  // expand_16
        };

        graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info),
                            get_weights_accessor(data_path, common_params.image))
              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "Conv_weights.npy"),
                  get_weights_accessor(data_path, "Conv_bias.npy"),
                  PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
                  1, conv_weights_quant_info.at(0), mid_quant_info)
              .set_name("Conv")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name("Conv/Relu6")
              << DepthwiseConvolutionLayer(3U, 3U,
                                           get_weights_accessor(data_path, "expanded_conv_depthwise_depthwise_weights.npy"),
                                           get_weights_accessor(data_path, "expanded_conv_depthwise_depthwise_biases.npy"),
                                           PadStrideInfo(1, 1, 1, 1), 1, dwc_q.at(0))
              .set_name("expanded_conv/depthwise/depthwise")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name("expanded_conv/depthwise/Relu6")
              << ConvolutionLayer(1U, 1U, 16U,
                                  get_weights_accessor(data_path, "expanded_conv_project_weights.npy"),
                                  get_weights_accessor(data_path, "expanded_conv_project_biases.npy"),
                                  PadStrideInfo(1, 1, 0, 0), 1, prwc_q.at(0))
              .set_name("expanded_conv/project/Conv2D");

        get_expanded_conv_qasymm8(data_path, "expanded_conv_1", IsResidual::No, 96U, 24U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                  pwc_q.at(1), dwc_q.at(1), prwc_q.at(1));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_2", IsResidual::Yes, 144U, 24U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(2), dwc_q.at(2), prwc_q.at(2));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_3", IsResidual::No, 144U, 32U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                  pwc_q.at(3), dwc_q.at(3), prwc_q.at(3));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_4", IsResidual::Yes, 192U, 32U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(4), dwc_q.at(4), prwc_q.at(4));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_5", IsResidual::Yes, 192U, 32U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(5), dwc_q.at(5), prwc_q.at(5));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_6", IsResidual::No, 192U, 64U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                  pwc_q.at(6), dwc_q.at(6), prwc_q.at(6));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_7", IsResidual::Yes, 384U, 64U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(7), dwc_q.at(7), prwc_q.at(7));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_8", IsResidual::Yes, 384U, 64U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(8), dwc_q.at(8), prwc_q.at(8));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_9", IsResidual::Yes, 384U, 64U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(9), dwc_q.at(9), prwc_q.at(9));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_10", IsResidual::No, 384U, 96U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(10), dwc_q.at(10), prwc_q.at(10));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_11", IsResidual::Yes, 576U, 96U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(11), dwc_q.at(11), prwc_q.at(11));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_12", IsResidual::Yes, 576U, 96U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(12), dwc_q.at(12), prwc_q.at(12));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_13", IsResidual::No, 576U, 160U, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL),
                                  pwc_q.at(13), dwc_q.at(13), prwc_q.at(13));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_14", IsResidual::Yes, 960U, 160U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(14), dwc_q.at(14), prwc_q.at(14));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_15", IsResidual::Yes, 960U, 160U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(15), dwc_q.at(15), prwc_q.at(15));
        get_expanded_conv_qasymm8(data_path, "expanded_conv_16", IsResidual::No, 960U, 320U, PadStrideInfo(1, 1, 1, 1), pwc_q.at(16), dwc_q.at(16), prwc_q.at(16));

        graph << ConvolutionLayer(1U, 1U, 1280U,
                                  get_weights_accessor(data_path, "Conv_1_weights.npy"),
                                  get_weights_accessor(data_path, "Conv_1_biases.npy"),
                                  PadStrideInfo(1, 1, 0, 0), 1, conv_weights_quant_info.at(1))
              .set_name("Conv_1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name("Conv_1/Relu6")
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("Logits/AvgPool")
              << ConvolutionLayer(1U, 1U, 1001U,
                                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy"),
                                  get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_biases.npy"),
                                  PadStrideInfo(1, 1, 0, 0), 1, conv_weights_quant_info.at(2))
              .set_name("Logits/Conv2d_1c_1x1");
    }

    void get_expanded_conv_qasymm8(const std::string &data_path, std::string &&param_path, IsResidual is_residual,
                                   unsigned int input_channels, unsigned int output_channels,
                                   PadStrideInfo           dwc_pad_stride_info,
                                   const QuantizationInfo &pwi, const QuantizationInfo &dwi, const QuantizationInfo &pji)
    {
        std::string total_path = param_path + "_";

        SubStream left(graph);
        left << ConvolutionLayer(1U, 1U, input_channels,
                                 get_weights_accessor(data_path, total_path + "project_weights.npy"),
                                 get_weights_accessor(data_path, total_path + "project_biases.npy"),
                                 PadStrideInfo(1, 1, 0, 0), 1, pwi)
             .set_name(param_path + "/Conv2D")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name(param_path + "/Conv2D/Relu6")
             << DepthwiseConvolutionLayer(3U, 3U,
                                          get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy"),
                                          get_weights_accessor(data_path, total_path + "depthwise_depthwise_biases.npy"),
                                          dwc_pad_stride_info, 1, dwi)
             .set_name(param_path + "/depthwise/depthwise")
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)).set_name(param_path + "/depthwise/Relu6")
             << ConvolutionLayer(1U, 1U, output_channels,
                                 get_weights_accessor(data_path, total_path + "project_weights.npy"),
                                 get_weights_accessor(data_path, total_path + "project_biases.npy"),
                                 PadStrideInfo(1, 1, 0, 0), 1, pji)
             .set_name(param_path + "/project/Conv2D");

        if(is_residual == IsResidual::Yes)
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
 * Provenance: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
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
