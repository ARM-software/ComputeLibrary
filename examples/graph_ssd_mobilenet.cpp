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

/** Example demonstrating how to implement MobileNetSSD's network using the Compute Library's graph API */
class GraphSSDMobilenetExample : public Example
{
public:
    GraphSSDMobilenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "MobileNetSSD")
    {
        // Add topk option
        keep_topk_opt = cmd_parser.add_option<SimpleOption<int>>("topk", 100);
        keep_topk_opt->set_help("Top k detections results per image.");
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
              << DepthwiseConvolutionMethod::Optimized3x3 // FIXME(COMPMID-1073): Add heuristics to automatically call the optimized 3x3 method
              << common_params.fast_math_hint;

        // Create core graph
        std::string model_path = "/cnn_data/ssd_mobilenet_model/";

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 127.5f, 127.5f, 127.5f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb, 0.007843f);

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
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

        conv_11 << get_node_A(conv_11, data_path, "conv1", 64, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv2", 128, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv3", 128, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv4", 256, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv5", 256, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv6", 512, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv7", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv8", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv9", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv10", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_11 << get_node_A(conv_11, data_path, "conv11", 512, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));

        SubStream conv_13(conv_11);
        conv_13 << get_node_A(conv_11, data_path, "conv12", 1024, PadStrideInfo(2, 2, 1, 1), PadStrideInfo(1, 1, 0, 0));
        conv_13 << get_node_A(conv_13, data_path, "conv13", 1024, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));

        SubStream conv_14(conv_13);
        conv_14 << get_node_B(conv_13, data_path, "conv14", 512, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        SubStream conv_15(conv_14);
        conv_15 << get_node_B(conv_14, data_path, "conv15", 256, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        SubStream conv_16(conv_15);
        conv_16 << get_node_B(conv_15, data_path, "conv16", 256, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        SubStream conv_17(conv_16);
        conv_17 << get_node_B(conv_16, data_path, "conv17", 128, PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 1, 1));

        //mbox_loc
        SubStream conv_11_mbox_loc(conv_11);
        conv_11_mbox_loc << get_node_C(conv_11, data_path, "conv11_mbox_loc", 12, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_13_mbox_loc(conv_13);
        conv_13_mbox_loc << get_node_C(conv_13, data_path, "conv13_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_14_2_mbox_loc(conv_14);
        conv_14_2_mbox_loc << get_node_C(conv_14, data_path, "conv14_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_15_2_mbox_loc(conv_15);
        conv_15_2_mbox_loc << get_node_C(conv_15, data_path, "conv15_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_16_2_mbox_loc(conv_16);
        conv_16_2_mbox_loc << get_node_C(conv_16, data_path, "conv16_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_17_2_mbox_loc(conv_17);
        conv_17_2_mbox_loc << get_node_C(conv_17, data_path, "conv17_2_mbox_loc", 24, PadStrideInfo(1, 1, 0, 0));

        SubStream mbox_loc(graph);
        mbox_loc << ConcatLayer(std::move(conv_11_mbox_loc), std::move(conv_13_mbox_loc), conv_14_2_mbox_loc, std::move(conv_15_2_mbox_loc),
                                std::move(conv_16_2_mbox_loc), std::move(conv_17_2_mbox_loc));

        //mbox_conf
        SubStream conv_11_mbox_conf(conv_11);
        conv_11_mbox_conf << get_node_C(conv_11, data_path, "conv11_mbox_conf", 63, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_13_mbox_conf(conv_13);
        conv_13_mbox_conf << get_node_C(conv_13, data_path, "conv13_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_14_2_mbox_conf(conv_14);
        conv_14_2_mbox_conf << get_node_C(conv_14, data_path, "conv14_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_15_2_mbox_conf(conv_15);
        conv_15_2_mbox_conf << get_node_C(conv_15, data_path, "conv15_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_16_2_mbox_conf(conv_16);
        conv_16_2_mbox_conf << get_node_C(conv_16, data_path, "conv16_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

        SubStream conv_17_2_mbox_conf(conv_17);
        conv_17_2_mbox_conf << get_node_C(conv_17, data_path, "conv17_2_mbox_conf", 126, PadStrideInfo(1, 1, 0, 0));

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
                          (common_params.data_layout == DataLayout::NCHW) ? DataLayoutDimension::WIDTH : DataLayoutDimension::CHANNEL,
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
        detection_ouput << OutputLayer(get_detection_output_accessor(common_params, { tensor_shape }));

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

    ConcatLayer get_node_A(IStream &master_graph, const std::string &data_path, std::string &&param_path,
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

    ConcatLayer get_node_B(IStream &master_graph, const std::string &data_path, std::string &&param_path,
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

    ConcatLayer get_node_C(IStream &master_graph, const std::string &data_path, std::string &&param_path,
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
