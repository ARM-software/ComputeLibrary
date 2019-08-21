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

/** Example demonstrating how to implement MobileNetSSD's network using the Compute Library's graph API */
class GraphSSDMobilenetExample : public Example
{
public:
    GraphSSDMobilenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "MobileNetSSD")
    {
        // Add topk option
        keep_topk_opt = cmd_parser.add_option<SimpleOption<int>>("topk", 100);
        keep_topk_opt->set_help("Top k detections results per image. Used for data type F32.");
        // Add output option
        detection_boxes_opt = cmd_parser.add_option<SimpleOption<std::string>>("detection_boxes_opt", "");
        detection_boxes_opt->set_help("Filename containing the reference values for the graph output detection_boxes. Used for data type QASYMM8.");
        detection_classes_opt = cmd_parser.add_option<SimpleOption<std::string>>("detection_classes_opt", "");
        detection_classes_opt->set_help("Filename containing the reference values for the output detection_classes. Used for data type QASYMM8.");
        detection_scores_opt = cmd_parser.add_option<SimpleOption<std::string>>("detection_scores_opt", "");
        detection_scores_opt->set_help("Filename containing the reference values for the output detection_scores. Used for data type QASYMM8.");
        num_detections_opt = cmd_parser.add_option<SimpleOption<std::string>>("num_detections_opt", "");
        num_detections_opt->set_help("Filename containing the reference values for the output num_detections. Used with datatype QASYMM8.");
    }
    GraphSSDMobilenetExample(const GraphSSDMobilenetExample &) = delete;
    GraphSSDMobilenetExample &operator=(const GraphSSDMobilenetExample &) = delete;
    GraphSSDMobilenetExample(GraphSSDMobilenetExample &&)                 = default; // NOLINT
    GraphSSDMobilenetExample &operator=(GraphSSDMobilenetExample &&) = default;      // NOLINT
    ~GraphSSDMobilenetExample() override                             = default;
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

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
        const TensorShape tensor_shape     = permute_shape(TensorShape(300, 300, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
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
            create_graph_qasymm(input_descriptor);
        }

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
    SimpleOption<int> *keep_topk_opt{ nullptr };
    CommonGraphParams  common_params;
    Stream             graph;

    SimpleOption<std::string> *detection_boxes_opt{ nullptr };
    SimpleOption<std::string> *detection_classes_opt{ nullptr };
    SimpleOption<std::string> *detection_scores_opt{ nullptr };
    SimpleOption<std::string> *num_detections_opt{ nullptr };

    ConcatLayer get_node_A_float(IStream &master_graph, const std::string &data_path, std::string &&param_path,
                                 unsigned int  conv_filt,
                                 PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
    {
        const std::string total_path = param_path + "_";
        SubStream         sg(master_graph);

        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "dw_w.npy"),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               dwc_pad_stride_info)
           .set_name(param_path + "/dw")
           << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "dw_bn_mean.npy"),
                                      get_weights_accessor(data_path, total_path + "dw_bn_var.npy"),
                                      get_weights_accessor(data_path, total_path + "dw_scale_w.npy"),
                                      get_weights_accessor(data_path, total_path + "dw_scale_b.npy"), 0.00001f)
           .set_name(param_path + "/dw/bn")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "dw/relu")

           << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "w.npy"),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               conv_pad_stride_info)
           .set_name(param_path + "/pw")
           << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "bn_mean.npy"),
                                      get_weights_accessor(data_path, total_path + "bn_var.npy"),
                                      get_weights_accessor(data_path, total_path + "scale_w.npy"),
                                      get_weights_accessor(data_path, total_path + "scale_b.npy"), 0.00001f)
           .set_name(param_path + "/pw/bn")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(param_path + "pw/relu");

        return ConcatLayer(std::move(sg));
    }

    ConcatLayer get_node_B_float(IStream &master_graph, const std::string &data_path, std::string &&param_path,
                                 unsigned int  conv_filt,
                                 PadStrideInfo conv_pad_stride_info_1, PadStrideInfo conv_pad_stride_info_2)
    {
        const std::string total_path = param_path + "_";
        SubStream         sg(master_graph);

        sg << ConvolutionLayer(
               1, 1, conv_filt / 2,
               get_weights_accessor(data_path, total_path + "1_w.npy"),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               conv_pad_stride_info_1)
           .set_name(total_path + "1/conv")
           << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "1_bn_mean.npy"),
                                      get_weights_accessor(data_path, total_path + "1_bn_var.npy"),
                                      get_weights_accessor(data_path, total_path + "1_scale_w.npy"),
                                      get_weights_accessor(data_path, total_path + "1_scale_b.npy"), 0.00001f)
           .set_name(total_path + "1/bn")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(total_path + "1/relu");

        sg << ConvolutionLayer(
               3, 3, conv_filt,
               get_weights_accessor(data_path, total_path + "2_w.npy"),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               conv_pad_stride_info_2)
           .set_name(total_path + "2/conv")
           << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "2_bn_mean.npy"),
                                      get_weights_accessor(data_path, total_path + "2_bn_var.npy"),
                                      get_weights_accessor(data_path, total_path + "2_scale_w.npy"),
                                      get_weights_accessor(data_path, total_path + "2_scale_b.npy"), 0.00001f)
           .set_name(total_path + "2/bn")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(total_path + "2/relu");

        return ConcatLayer(std::move(sg));
    }

    ConcatLayer get_node_C_float(IStream &master_graph, const std::string &data_path, std::string &&param_path,
                                 unsigned int conv_filt, PadStrideInfo conv_pad_stride_info)
    {
        const std::string total_path = param_path + "_";
        SubStream         sg(master_graph);
        sg << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "w.npy"),
               get_weights_accessor(data_path, total_path + "b.npy"),
               conv_pad_stride_info)
           .set_name(param_path + "/conv");
        if(common_params.data_layout == DataLayout::NCHW)
        {
            sg << PermuteLayer(PermutationVector(2U, 0U, 1U), DataLayout::NHWC).set_name(param_path + "/perm");
        }
        sg << FlattenLayer().set_name(param_path + "/flat");

        return ConcatLayer(std::move(sg));
    }

    void create_graph_float(TensorDescriptor &input_descriptor)
    {
        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 127.5f, 127.5f, 127.5f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb, true, 0.007843f);

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += "/cnn_data/ssd_mobilenet_model/";
        }

        graph << InputLayer(input_descriptor,
                            get_input_accessor(common_params, std::move(preprocessor)));

        SubStream conv_11(graph);
        conv_11 << ConvolutionLayer(
                    3U, 3U, 32U,
                    get_weights_accessor(data_path, "conv0_w.npy"),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(2, 2, 1, 1))
                .set_name("conv0");
        conv_11 << BatchNormalizationLayer(get_weights_accessor(data_path, "conv0_bn_mean.npy"),
                                           get_weights_accessor(data_path, "conv0_bn_var.npy"),
                                           get_weights_accessor(data_path, "conv0_scale_w.npy"),
                                           get_weights_accessor(data_path, "conv0_scale_b.npy"), 0.00001f)
                .set_name("conv0/bn")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv0/relu");

        conv_11 << get_node_A_float(conv_11, data_path, "conv1", 64, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv2", 128, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv3", 128, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv4", 256, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv5", 256, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv6", 512, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv7", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv8", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv9", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv10", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A_float(conv_11, data_path, "conv11", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));

        SubStream conv_13(conv_11);
        conv_13 << get_node_A_float(conv_11, data_path, "conv12", 1024, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_13 << get_node_A_float(conv_13, data_path, "conv13", 1024, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));

        SubStream conv_14(conv_13);
        conv_14 << get_node_B_float(conv_13, data_path, "conv14", 512, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        SubStream conv_15(conv_14);
        conv_15 << get_node_B_float(conv_14, data_path, "conv15", 256, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        SubStream conv_16(conv_15);
        conv_16 << get_node_B_float(conv_15, data_path, "conv16", 256, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        SubStream conv_17(conv_16);
        conv_17 << get_node_B_float(conv_16, data_path, "conv17", 128, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        //mbox_loc
        SubStream conv_11_mbox_loc(conv_11);
        conv_11_mbox_loc << get_node_C_float(conv_11, data_path, "conv11_mbox_loc", 12, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_13_mbox_loc(conv_13);
        conv_13_mbox_loc << get_node_C_float(conv_13, data_path, "conv13_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_14_2_mbox_loc(conv_14);
        conv_14_2_mbox_loc << get_node_C_float(conv_14, data_path, "conv14_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_15_2_mbox_loc(conv_15);
        conv_15_2_mbox_loc << get_node_C_float(conv_15, data_path, "conv15_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_16_2_mbox_loc(conv_16);
        conv_16_2_mbox_loc << get_node_C_float(conv_16, data_path, "conv16_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_17_2_mbox_loc(conv_17);
        conv_17_2_mbox_loc << get_node_C_float(conv_17, data_path, "conv17_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream mbox_loc(graph);
        mbox_loc << ConcatLayer(std::move(conv_11_mbox_loc), std::move(conv_13_mbox_loc), conv_14_2_mbox_loc, std::move(conv_15_2_mbox_loc),
                                std::move(conv_16_2_mbox_loc), std::move(conv_17_2_mbox_loc));

        //mbox_conf
        SubStream conv_11_mbox_conf(conv_11);
        conv_11_mbox_conf << get_node_C_float(conv_11, data_path, "conv11_mbox_conf", 63, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_13_mbox_conf(conv_13);
        conv_13_mbox_conf << get_node_C_float(conv_13, data_path, "conv13_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_14_2_mbox_conf(conv_14);
        conv_14_2_mbox_conf << get_node_C_float(conv_14, data_path, "conv14_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_15_2_mbox_conf(conv_15);
        conv_15_2_mbox_conf << get_node_C_float(conv_15, data_path, "conv15_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_16_2_mbox_conf(conv_16);
        conv_16_2_mbox_conf << get_node_C_float(conv_16, data_path, "conv16_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_17_2_mbox_conf(conv_17);
        conv_17_2_mbox_conf << get_node_C_float(conv_17, data_path, "conv17_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream mbox_conf(graph);
        mbox_conf << ConcatLayer(std::move(conv_11_mbox_conf), std::move(conv_13_mbox_conf), std::move(conv_14_2_mbox_conf),
                                 std::move(conv_15_2_mbox_conf), std::move(conv_16_2_mbox_conf), std::move(conv_17_2_mbox_conf));
        mbox_conf << ReshapeLayer(TensorShape(21U, 1917U)).set_name("mbox_conf/reshape");
        mbox_conf << SoftmaxLayer().set_name("mbox_conf/softmax");
        mbox_conf << FlattenLayer().set_name("mbox_conf/flat");

        const std::vector<float> priorbox_variances     = { 0.1f, 0.1f, 0.2f, 0.2f };
        const float              priorbox_offset        = 0.5f;
        const std::vector<float> priorbox_aspect_ratios = { 2.f, 3.f };

        //mbox_priorbox branch
        SubStream conv_11_mbox_priorbox(conv_11);

        conv_11_mbox_priorbox << PriorBoxLayer(SubStream(graph),
                                               PriorBoxLayerInfo({ 60.f }, priorbox_variances, priorbox_offset, true, false, {}, { 2.f }))
                              .set_name("conv11/priorbox");

        SubStream conv_13_mbox_priorbox(conv_13);
        conv_13_mbox_priorbox << PriorBoxLayer(SubStream(graph),
                                               PriorBoxLayerInfo({ 105.f }, priorbox_variances, priorbox_offset, true, false, { 150.f }, priorbox_aspect_ratios))
                              .set_name("conv13/priorbox");

        SubStream conv_14_2_mbox_priorbox(conv_14);
        conv_14_2_mbox_priorbox << PriorBoxLayer(SubStream(graph),
                                                 PriorBoxLayerInfo({ 150.f }, priorbox_variances, priorbox_offset, true, false, { 195.f }, priorbox_aspect_ratios))
                                .set_name("conv14/priorbox");

        SubStream conv_15_2_mbox_priorbox(conv_15);
        conv_15_2_mbox_priorbox << PriorBoxLayer(SubStream(graph),
                                                 PriorBoxLayerInfo({ 195.f }, priorbox_variances, priorbox_offset, true, false, { 240.f }, priorbox_aspect_ratios))
                                .set_name("conv15/priorbox");

        SubStream conv_16_2_mbox_priorbox(conv_16);
        conv_16_2_mbox_priorbox << PriorBoxLayer(SubStream(graph),
                                                 PriorBoxLayerInfo({ 240.f }, priorbox_variances, priorbox_offset, true, false, { 285.f }, priorbox_aspect_ratios))
                                .set_name("conv16/priorbox");

        SubStream conv_17_2_mbox_priorbox(conv_17);
        conv_17_2_mbox_priorbox << PriorBoxLayer(SubStream(graph),
                                                 PriorBoxLayerInfo({ 285.f }, priorbox_variances, priorbox_offset, true, false, { 300.f }, priorbox_aspect_ratios))
                                .set_name("conv17/priorbox");

        SubStream mbox_priorbox(graph);

        mbox_priorbox << ConcatLayer(
                          (common_params.data_layout == DataLayout::NCHW) ? arm_compute::graph::descriptors::ConcatLayerDescriptor(DataLayoutDimension::WIDTH) : arm_compute::graph::descriptors::ConcatLayerDescriptor(
                              DataLayoutDimension::CHANNEL),
                          std::move(conv_11_mbox_priorbox), std::move(conv_13_mbox_priorbox), std::move(conv_14_2_mbox_priorbox),
                          std::move(conv_15_2_mbox_priorbox), std::move(conv_16_2_mbox_priorbox), std::move(conv_17_2_mbox_priorbox));

        const int                          num_classes         = 21;
        const bool                         share_location      = true;
        const DetectionOutputLayerCodeType detection_type      = DetectionOutputLayerCodeType::CENTER_SIZE;
        const int                          keep_top_k          = keep_topk_opt->value();
        const float                        nms_threshold       = 0.45f;
        const int                          label_id_background = 0;
        const float                        conf_thrs           = 0.25f;
        const int                          top_k               = 100;

        SubStream detection_ouput(mbox_loc);
        detection_ouput << DetectionOutputLayer(std::move(mbox_conf), std::move(mbox_priorbox),
                                                DetectionOutputLayerInfo(num_classes, share_location, detection_type, keep_top_k, nms_threshold, top_k, label_id_background, conf_thrs));
        detection_ouput << OutputLayer(get_detection_output_accessor(common_params, { input_descriptor.shape }));
    }

    ConcatLayer get_node_A_qasymm(IStream &master_graph, const std::string &data_path, std::string &&param_path,
                                  unsigned int  conv_filt,
                                  PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info,
                                  std::pair<QuantizationInfo, QuantizationInfo> depth_quant_info, std::pair<QuantizationInfo, QuantizationInfo> point_quant_info)
    {
        const std::string total_path = param_path + "_";
        SubStream         sg(master_graph);

        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "dw_w.npy"),
               get_weights_accessor(data_path, total_path + "dw_b.npy"),
               dwc_pad_stride_info, 1, depth_quant_info.first, depth_quant_info.second)
           .set_name(param_path + "/dw")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(param_path + "/dw/relu6");

        sg << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "w.npy"),
               get_weights_accessor(data_path, total_path + "b.npy"),
               conv_pad_stride_info, 1, point_quant_info.first, point_quant_info.second)
           .set_name(param_path + "/pw")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(param_path + "/pw/relu6");

        return ConcatLayer(std::move(sg));
    }

    ConcatLayer get_node_B_qasymm(IStream &master_graph, const std::string &data_path, std::string &&param_path,
                                  unsigned int  conv_filt,
                                  PadStrideInfo conv_pad_stride_info_1x1, PadStrideInfo conv_pad_stride_info_3x3,
                                  const std::pair<QuantizationInfo, QuantizationInfo> quant_info_1x1, const std::pair<QuantizationInfo, QuantizationInfo> quant_info_3x3)
    {
        const std::string total_path = param_path + "_";
        SubStream         sg(master_graph);

        sg << ConvolutionLayer(
               1, 1, conv_filt / 2,
               get_weights_accessor(data_path, total_path + "1x1_w.npy"),
               get_weights_accessor(data_path, total_path + "1x1_b.npy"),
               conv_pad_stride_info_1x1, 1, quant_info_1x1.first, quant_info_1x1.second)
           .set_name(total_path + "1x1/conv")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "1x1/conv/relu6");

        sg << ConvolutionLayer(
               3, 3, conv_filt,
               get_weights_accessor(data_path, total_path + "3x3_w.npy"),
               get_weights_accessor(data_path, total_path + "3x3_b.npy"),
               conv_pad_stride_info_3x3, 1, quant_info_3x3.first, quant_info_3x3.second)
           .set_name(total_path + "3x3/conv")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "3x3/conv/relu6");

        return ConcatLayer(std::move(sg));
    }

    ConcatLayer get_node_C_qasymm(IStream &master_graph, const std::string &data_path, std::string &&param_path,
                                  unsigned int conv_filt, PadStrideInfo               conv_pad_stride_info,
                                  const std::pair<QuantizationInfo, QuantizationInfo> quant_info, TensorShape reshape_shape)
    {
        const std::string total_path = param_path + "_";
        SubStream         sg(master_graph);
        sg << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "w.npy"),
               get_weights_accessor(data_path, total_path + "b.npy"),
               conv_pad_stride_info, 1, quant_info.first, quant_info.second)
           .set_name(param_path + "/conv");
        if(common_params.data_layout == DataLayout::NCHW)
        {
            sg << PermuteLayer(PermutationVector(2U, 0U, 1U), DataLayout::NHWC);
        }
        sg << ReshapeLayer(reshape_shape).set_name(param_path + "/reshape");

        return ConcatLayer(std::move(sg));
    }

    void create_graph_qasymm(TensorDescriptor &input_descriptor)
    {
        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += "/cnn_data/ssd_mobilenet_qasymm8_model/";
        }

        // Quantization info are saved as pair for each (pointwise/depthwise) convolution layer: <weight_quant_info, output_quant_info>
        const std::vector<std::pair<QuantizationInfo, QuantizationInfo>> conv_quant_info =
        {
            { QuantizationInfo(0.03624850884079933f, 163), QuantizationInfo(0.22219789028167725f, 113) },   // conv0
            { QuantizationInfo(0.0028752065263688564f, 113), QuantizationInfo(0.05433657020330429f, 128) }, // conv13_2_1_1
            { QuantizationInfo(0.0014862528769299388f, 125), QuantizationInfo(0.05037643015384674f, 131) }, // conv13_2_3_3
            { QuantizationInfo(0.00233650766313076f, 113), QuantizationInfo(0.04468846693634987f, 126) },   // conv13_3_1_1
            { QuantizationInfo(0.002501056529581547f, 120), QuantizationInfo(0.06026708707213402f, 111) },  // conv13_3_3_3
            { QuantizationInfo(0.002896666992455721f, 121), QuantizationInfo(0.037775348871946335f, 117) }, // conv13_4_1_1
            { QuantizationInfo(0.0023875406477600336f, 122), QuantizationInfo(0.03881589323282242f, 108) }, // conv13_4_3_3
            { QuantizationInfo(0.0022081052884459496f, 77), QuantizationInfo(0.025450613349676132f, 125) }, // conv13_5_1_1
            { QuantizationInfo(0.00604657270014286f, 121), QuantizationInfo(0.033533502370119095f, 109) }   // conv13_5_3_3
        };

        const std::vector<std::pair<QuantizationInfo, QuantizationInfo>> depth_quant_info =
        {
            { QuantizationInfo(0.03408717364072f, 131), QuantizationInfo(0.29286590218544006f, 108) },     // dwsc1
            { QuantizationInfo(0.027518004179000854f, 107), QuantizationInfo(0.20796941220760345, 117) },  // dwsc2
            { QuantizationInfo(0.052489638328552246f, 85), QuantizationInfo(0.4303881824016571f, 142) },   // dwsc3
            { QuantizationInfo(0.016570359468460083f, 79), QuantizationInfo(0.10512150079011917f, 116) },  // dwsc4
            { QuantizationInfo(0.060739465057849884f, 65), QuantizationInfo(0.15331414341926575f, 94) },   // dwsc5
            { QuantizationInfo(0.01324534136801958f, 124), QuantizationInfo(0.13010895252227783f, 153) },  // dwsc6
            { QuantizationInfo(0.032326459884643555f, 124), QuantizationInfo(0.11565316468477249, 156) },  // dwsc7
            { QuantizationInfo(0.029948478564620018f, 155), QuantizationInfo(0.11413891613483429f, 146) }, // dwsc8
            { QuantizationInfo(0.028054025024175644f, 129), QuantizationInfo(0.1142905130982399f, 140) },  // dwsc9
            { QuantizationInfo(0.025204822421073914f, 129), QuantizationInfo(0.14668069779872894f, 149) }, // dwsc10
            { QuantizationInfo(0.019332280382514f, 110), QuantizationInfo(0.1480235457420349f, 91) },      // dwsc11
            { QuantizationInfo(0.0319712869822979f, 88), QuantizationInfo(0.10424695909023285f, 117) },    // dwsc12
            { QuantizationInfo(0.04378943517804146f, 164), QuantizationInfo(0.23176774382591248f, 138) }   // dwsc13
        };

        const std::vector<std::pair<QuantizationInfo, QuantizationInfo>> point_quant_info =
        {
            { QuantizationInfo(0.028777318075299263f, 144), QuantizationInfo(0.2663874328136444f, 121) },  // pw1
            { QuantizationInfo(0.015796702355146408f, 127), QuantizationInfo(0.1739964485168457f, 111) },  // pw2
            { QuantizationInfo(0.009349990636110306f, 127), QuantizationInfo(0.1805974692106247f, 104) },  // pw3
            { QuantizationInfo(0.012920888140797615f, 106), QuantizationInfo(0.1205204650759697f, 100) },  // pw4
            { QuantizationInfo(0.008119508624076843f, 145), QuantizationInfo(0.12272439152002335f, 97) },  // pw5
            { QuantizationInfo(0.0070041813887655735f, 115), QuantizationInfo(0.0947074219584465f, 101) }, // pw6
            { QuantizationInfo(0.004827278666198254f, 115), QuantizationInfo(0.0842885747551918f, 110) },  // pw7
            { QuantizationInfo(0.004755120258778334f, 128), QuantizationInfo(0.08283159881830215f, 116) }, // pw8
            { QuantizationInfo(0.007527193054556847f, 142), QuantizationInfo(0.12555131316184998f, 137) }, // pw9
            { QuantizationInfo(0.006050156895071268f, 109), QuantizationInfo(0.10871313512325287f, 124) }, // pw10
            { QuantizationInfo(0.00490700313821435f, 127), QuantizationInfo(0.10364262014627457f, 140) },  // pw11
            { QuantizationInfo(0.006063731852918863, 124), QuantizationInfo(0.11241862177848816f, 125) },  // pw12
            { QuantizationInfo(0.007901716977357864f, 139), QuantizationInfo(0.49889302253723145f, 141) }  // pw13
        };

        // Quantization info taken from the TfLite SSD MobileNet example
        const QuantizationInfo in_quant_info = QuantizationInfo(0.0078125f, 128);

        // Create core graph
        graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info),
                            get_weights_accessor(data_path, common_params.image, DataLayout::NHWC));
        graph << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "conv0_w.npy"),
                  get_weights_accessor(data_path, "conv0_b.npy"),
                  PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::CEIL), 1, conv_quant_info.at(0).first, conv_quant_info.at(0).second)
              .set_name("conv0");
        graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name("conv0/relu");
        graph << get_node_A_qasymm(graph, data_path, "conv1", 64U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(0),
                                   point_quant_info.at(0));
        graph << get_node_A_qasymm(graph, data_path, "conv2", 128U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(1),
                                   point_quant_info.at(1));
        graph << get_node_A_qasymm(graph, data_path, "conv3", 128U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(2),
                                   point_quant_info.at(2));
        graph << get_node_A_qasymm(graph, data_path, "conv4", 256U, PadStrideInfo(2U, 2U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(3),
                                   point_quant_info.at(3));
        graph << get_node_A_qasymm(graph, data_path, "conv5", 256U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(4),
                                   point_quant_info.at(4));
        graph << get_node_A_qasymm(graph, data_path, "conv6", 512U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(5),
                                   point_quant_info.at(5));
        graph << get_node_A_qasymm(graph, data_path, "conv7", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(6),
                                   point_quant_info.at(6));
        graph << get_node_A_qasymm(graph, data_path, "conv8", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(7),
                                   point_quant_info.at(7));
        graph << get_node_A_qasymm(graph, data_path, "conv9", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(8),
                                   point_quant_info.at(8));
        graph << get_node_A_qasymm(graph, data_path, "conv10", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(9),
                                   point_quant_info.at(9));
        graph << get_node_A_qasymm(graph, data_path, "conv11", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(10),
                                   point_quant_info.at(10));

        SubStream conv_13(graph);
        conv_13 << get_node_A_qasymm(graph, data_path, "conv12", 1024U, PadStrideInfo(2U, 2U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(11),
                                     point_quant_info.at(11));
        conv_13 << get_node_A_qasymm(conv_13, data_path, "conv13", 1024U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), PadStrideInfo(1U, 1U, 0U, 0U), depth_quant_info.at(12),
                                     point_quant_info.at(12));
        SubStream conv_14(conv_13);
        conv_14 << get_node_B_qasymm(conv_13, data_path, "conv13_2", 512U, PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::CEIL), conv_quant_info.at(1),
                                     conv_quant_info.at(2));
        SubStream conv_15(conv_14);
        conv_15 << get_node_B_qasymm(conv_14, data_path, "conv13_3", 256U, PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(2U, 2U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), conv_quant_info.at(3),
                                     conv_quant_info.at(4));
        SubStream conv_16(conv_15);
        conv_16 << get_node_B_qasymm(conv_15, data_path, "conv13_4", 256U, PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(2U, 2U, 1U, 1U, 1U, 1U, DimensionRoundingType::CEIL), conv_quant_info.at(5),
                                     conv_quant_info.at(6));
        SubStream conv_17(conv_16);
        conv_17 << get_node_B_qasymm(conv_16, data_path, "conv13_5", 128U, PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::CEIL), conv_quant_info.at(7),
                                     conv_quant_info.at(8));

        // box_predictor
        const std::vector<std::pair<QuantizationInfo, QuantizationInfo>> box_enc_pred_quant_info =
        {
            { QuantizationInfo(0.005202020984143019f, 136), QuantizationInfo(0.08655580133199692f, 183) },   // boxpredictor0_bep
            { QuantizationInfo(0.003121797926723957f, 132), QuantizationInfo(0.03218776360154152f, 140) },   // boxpredictor1_bep
            { QuantizationInfo(0.002995674265548587f, 130), QuantizationInfo(0.029072262346744537f, 125) },  // boxpredictor2_bep
            { QuantizationInfo(0.0023131705820560455f, 130), QuantizationInfo(0.026488754898309708f, 127) }, // boxpredictor3_bep
            { QuantizationInfo(0.0013905081432312727f, 132), QuantizationInfo(0.0199890099465847f, 137) },   // boxpredictor4_bep
            { QuantizationInfo(0.00216794665902853f, 121), QuantizationInfo(0.019798893481492996f, 151) }    // boxpredictor5_bep
        };

        const std::vector<TensorShape> box_reshape = // NHWC
        {
            TensorShape(4U, 1U, 1083U), // boxpredictor0_bep_reshape
            TensorShape(4U, 1U, 600U),  // boxpredictor1_bep_reshape
            TensorShape(4U, 1U, 150U),  // boxpredictor2_bep_reshape
            TensorShape(4U, 1U, 54U),   // boxpredictor3_bep_reshape
            TensorShape(4U, 1U, 24U),   // boxpredictor4_bep_reshape
            TensorShape(4U, 1U, 6U)     // boxpredictor5_bep_reshape
        };

        SubStream conv_11_box_enc_pre(graph);
        conv_11_box_enc_pre << get_node_C_qasymm(graph, data_path, "BoxPredictor_0_BEP", 12U, PadStrideInfo(1U, 1U, 0U, 0U), box_enc_pred_quant_info.at(0), box_reshape.at(0));

        SubStream conv_13_box_enc_pre(conv_13);
        conv_13_box_enc_pre << get_node_C_qasymm(conv_13, data_path, "BoxPredictor_1_BEP", 24U, PadStrideInfo(1U, 1U, 0U, 0U), box_enc_pred_quant_info.at(1), box_reshape.at(1));

        SubStream conv_14_2_box_enc_pre(conv_14);
        conv_14_2_box_enc_pre << get_node_C_qasymm(conv_14, data_path, "BoxPredictor_2_BEP", 24U, PadStrideInfo(1U, 1U, 0U, 0U), box_enc_pred_quant_info.at(2), box_reshape.at(2));

        SubStream conv_15_2_box_enc_pre(conv_15);
        conv_15_2_box_enc_pre << get_node_C_qasymm(conv_15, data_path, "BoxPredictor_3_BEP", 24U, PadStrideInfo(1U, 1U, 0U, 0U), box_enc_pred_quant_info.at(3), box_reshape.at(3));

        SubStream conv_16_2_box_enc_pre(conv_16);
        conv_16_2_box_enc_pre << get_node_C_qasymm(conv_16, data_path, "BoxPredictor_4_BEP", 24U, PadStrideInfo(1U, 1U, 0U, 0U), box_enc_pred_quant_info.at(4), box_reshape.at(4));

        SubStream conv_17_2_box_enc_pre(conv_17);
        conv_17_2_box_enc_pre << get_node_C_qasymm(conv_17, data_path, "BoxPredictor_5_BEP", 24U, PadStrideInfo(1U, 1U, 0U, 0U), box_enc_pred_quant_info.at(5), box_reshape.at(5));

        SubStream              box_enc_pre(graph);
        const QuantizationInfo bep_concate_qinfo = QuantizationInfo(0.08655580133199692f, 183);
        box_enc_pre << ConcatLayer(arm_compute::graph::descriptors::ConcatLayerDescriptor(DataLayoutDimension::HEIGHT, bep_concate_qinfo),
                                   std::move(conv_11_box_enc_pre), std::move(conv_13_box_enc_pre), conv_14_2_box_enc_pre, std::move(conv_15_2_box_enc_pre),
                                   std::move(conv_16_2_box_enc_pre), std::move(conv_17_2_box_enc_pre))
                    .set_name("BoxPredictor/concat");
        box_enc_pre << ReshapeLayer(TensorShape(4U, 1917U)).set_name("BoxPredictor/reshape");

        // class_predictor
        const std::vector<std::pair<QuantizationInfo, QuantizationInfo>> class_pred_quant_info =
        {
            { QuantizationInfo(0.002744135679677129f, 125), QuantizationInfo(0.05746262148022652f, 234) },   // boxpredictor0_cp
            { QuantizationInfo(0.0024326108396053314f, 80), QuantizationInfo(0.03764628246426582f, 217) },   // boxpredictor1_cp
            { QuantizationInfo(0.0013898586621508002f, 141), QuantizationInfo(0.034081317484378815f, 214) }, // boxpredictor2_cp
            { QuantizationInfo(0.0014176908880472183f, 133), QuantizationInfo(0.033889178186655045f, 215) }, // boxpredictor3_cp
            { QuantizationInfo(0.001090311910957098f, 125), QuantizationInfo(0.02646234817802906f, 230) },   // boxpredictor4_cp
            { QuantizationInfo(0.001134163816459477f, 115), QuantizationInfo(0.026926767081022263f, 218) }   // boxpredictor5_cp
        };

        const std::vector<TensorShape> class_reshape =
        {
            TensorShape(91U, 1083U), // boxpredictor0_cp_reshape
            TensorShape(91U, 600U),  // boxpredictor1_cp_reshape
            TensorShape(91U, 150U),  // boxpredictor2_cp_reshape
            TensorShape(91U, 54U),   // boxpredictor3_cp_reshape
            TensorShape(91U, 24U),   // boxpredictor4_cp_reshape
            TensorShape(91U, 6U)     // boxpredictor5_cp_reshape
        };

        SubStream conv_11_class_pre(graph);
        conv_11_class_pre << get_node_C_qasymm(graph, data_path, "BoxPredictor_0_CP", 273U, PadStrideInfo(1U, 1U, 0U, 0U), class_pred_quant_info.at(0), class_reshape.at(0));

        SubStream conv_13_class_pre(conv_13);
        conv_13_class_pre << get_node_C_qasymm(conv_13, data_path, "BoxPredictor_1_CP", 546U, PadStrideInfo(1U, 1U, 0U, 0U), class_pred_quant_info.at(1), class_reshape.at(1));

        SubStream conv_14_2_class_pre(conv_14);
        conv_14_2_class_pre << get_node_C_qasymm(conv_14, data_path, "BoxPredictor_2_CP", 546U, PadStrideInfo(1U, 1U, 0U, 0U), class_pred_quant_info.at(2), class_reshape.at(2));

        SubStream conv_15_2_class_pre(conv_15);
        conv_15_2_class_pre << get_node_C_qasymm(conv_15, data_path, "BoxPredictor_3_CP", 546U, PadStrideInfo(1U, 1U, 0U, 0U), class_pred_quant_info.at(3), class_reshape.at(3));

        SubStream conv_16_2_class_pre(conv_16);
        conv_16_2_class_pre << get_node_C_qasymm(conv_16, data_path, "BoxPredictor_4_CP", 546U, PadStrideInfo(1U, 1U, 0U, 0U), class_pred_quant_info.at(4), class_reshape.at(4));

        SubStream conv_17_2_class_pre(conv_17);
        conv_17_2_class_pre << get_node_C_qasymm(conv_17, data_path, "BoxPredictor_5_CP", 546U, PadStrideInfo(1U, 1U, 0U, 0U), class_pred_quant_info.at(5), class_reshape.at(5));

        const QuantizationInfo cp_concate_qinfo = QuantizationInfo(0.0584389753639698f, 230);
        SubStream              class_pred(graph);
        class_pred << ConcatLayer(
                       arm_compute::graph::descriptors::ConcatLayerDescriptor(DataLayoutDimension::WIDTH, cp_concate_qinfo),
                       std::move(conv_11_class_pre), std::move(conv_13_class_pre), std::move(conv_14_2_class_pre),
                       std::move(conv_15_2_class_pre), std::move(conv_16_2_class_pre), std::move(conv_17_2_class_pre))
                   .set_name("ClassPrediction/concat");

        const QuantizationInfo logistic_out_qinfo = QuantizationInfo(0.00390625f, 0);
        class_pred << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC), logistic_out_qinfo).set_name("ClassPrediction/logistic");

        const int   max_detections            = 10;
        const int   max_classes_per_detection = 1;
        const float nms_score_threshold       = 0.30000001192092896f;
        const float nms_iou_threshold         = 0.6000000238418579f;
        const int   num_classes               = 90;
        const float x_scale                   = 10.f;
        const float y_scale                   = 10.f;
        const float h_scale                   = 5.f;
        const float w_scale                   = 5.f;
        std::array<float, 4> scales = { y_scale, x_scale, w_scale, h_scale };
        const QuantizationInfo anchors_qinfo = QuantizationInfo(0.006453060545027256f, 0);

        SubStream detection_ouput(box_enc_pre);
        detection_ouput << DetectionPostProcessLayer(std::move(class_pred),
                                                     DetectionPostProcessLayerInfo(max_detections, max_classes_per_detection, nms_score_threshold, nms_iou_threshold, num_classes, scales),
                                                     get_weights_accessor(data_path, "anchors.npy"), anchors_qinfo)
                        .set_name("DetectionPostProcess");

        SubStream ouput_0(detection_ouput);
        ouput_0 << OutputLayer(get_npy_output_accessor(detection_boxes_opt->value(), TensorShape(4U, 10U), DataType::F32), 0);

        SubStream ouput_1(detection_ouput);
        ouput_1 << OutputLayer(get_npy_output_accessor(detection_classes_opt->value(), TensorShape(10U), DataType::F32), 1);

        SubStream ouput_2(detection_ouput);
        ouput_2 << OutputLayer(get_npy_output_accessor(detection_scores_opt->value(), TensorShape(10U), DataType::F32), 2);

        SubStream ouput_3(detection_ouput);
        ouput_3 << OutputLayer(get_npy_output_accessor(num_detections_opt->value(), TensorShape(1U), DataType::F32), 3);
    }
};

/** Main program for MobileNetSSD
 *
 * Model is based on:
 *      http://arxiv.org/abs/1512.02325
 *      SSD: Single Shot MultiBox Detector
 *      Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
 *
 * Provenance: https://github.com/chuanqi305/MobileNet-SSD
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphSSDMobilenetExample>(argc, argv);
}
