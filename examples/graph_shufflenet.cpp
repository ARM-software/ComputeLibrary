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

/** Example demonstrating how to implement ShuffleNet network using the Compute Library's graph API */
class ShuffleNetExample : public Example
{
public:
    ShuffleNetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "ShuffleNet")
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

        // Set default layout if needed (Single kernel grouped convolution not yet supported int NHWC)
        if(!common_opts.data_layout->is_set())
        {
            common_params.data_layout = DataLayout::NHWC;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;
        std::cout << "Model: Shufflenet_1_g4" << std::endl;

        // Create model path
        std::string model_path = "/cnn_data/shufflenet_model/";

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        // Create preprocessor
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>(0);

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false /* Do not convert to BGR */))
              << ConvolutionLayer(
                  3U, 3U, 24U,
                  get_weights_accessor(data_path, "conv3_0_w_0.npy", weights_layout),
                  get_weights_accessor(data_path, "conv3_0_b_0.npy", weights_layout),
                  PadStrideInfo(2, 2, 1, 1))
              .set_name("Conv1/convolution")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "conv3_0_bn_rm_0.npy"),
                  get_weights_accessor(data_path, "conv3_0_bn_riv_0.npy"),
                  get_weights_accessor(data_path, "conv3_0_bn_s_0.npy"),
                  get_weights_accessor(data_path, "conv3_0_bn_b_0.npy"),
                  1e-5f)
              .set_name("Conv1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("Conv1/Relu")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 1, 1))).set_name("pool1/MaxPool");

        // Stage 2
        add_residual_block(data_path, DataLayout::NCHW, 0U /* unit */, 112U /* depth */, 2U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 1U /* unit */, 136U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 2U /* unit */, 136U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 3U /* unit */, 136U /* depth */, 1U /* stride */);

        // Stage 3
        add_residual_block(data_path, DataLayout::NCHW, 4U /* unit */, 136U /* depth */, 2U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 5U /* unit */, 272U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 6U /* unit */, 272U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 7U /* unit */, 272U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 8U /* unit */, 272U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 9U /* unit */, 272U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 10U /* unit */, 272U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 11U /* unit */, 272U /* depth */, 1U /* stride */);

        // Stage 4
        add_residual_block(data_path, DataLayout::NCHW, 12U /* unit */, 272U /* depth */, 2U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 13U /* unit */, 544U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 14U /* unit */, 544U /* depth */, 1U /* stride */);
        add_residual_block(data_path, DataLayout::NCHW, 15U /* unit */, 544U /* depth */, 1U /* stride */);

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("predictions/AvgPool")
              << FlattenLayer().set_name("predictions/Reshape")
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "pred_w_0.npy", weights_layout),
                  get_weights_accessor(data_path, "pred_b_0.npy"))
              .set_name("predictions/FC")
              << SoftmaxLayer().set_name("predictions/Softmax")
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

    void add_residual_block(const std::string &data_path, DataLayout weights_layout,
                            unsigned int unit, unsigned int depth, unsigned int stride)
    {
        PadStrideInfo      dwc_info        = PadStrideInfo(1, 1, 1, 1);
        const unsigned int gconv_id        = unit * 2;
        const unsigned int num_groups      = 4;
        const std::string  unit_id_name    = arm_compute::support::cpp11::to_string(unit);
        const std::string  gconv_id_name   = arm_compute::support::cpp11::to_string(gconv_id);
        const std::string  gconv_id_1_name = arm_compute::support::cpp11::to_string(gconv_id + 1);
        const std::string  unit_name       = "unit" + unit_id_name;

        SubStream left_ss(graph);
        SubStream right_ss(graph);

        if(stride == 2)
        {
            right_ss << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(2, 2, 1, 1))).set_name(unit_name + "/pool_1/AveragePool");
            dwc_info = PadStrideInfo(2, 2, 1, 1);
        }

        left_ss << ConvolutionLayer(
                    1U, 1U, depth,
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_name + "_w_0.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(1, 1, 0, 0), num_groups)
                .set_name(unit_name + "/gconv1_" + gconv_id_name + "/convolution")
                << BatchNormalizationLayer(
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_name + "_bn_rm_0.npy"),
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_name + "_bn_riv_0.npy"),
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_name + "_bn_s_0.npy"),
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_name + "_bn_b_0.npy"),
                    1e-5f)
                .set_name(unit_name + "/gconv1_" + gconv_id_name + "/BatchNorm")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "/gconv1_" + gconv_id_name + "/Relu")
                << ChannelShuffleLayer(num_groups).set_name(unit_name + "/shuffle_0/ChannelShufle")
                << DepthwiseConvolutionLayer(
                    3U, 3U,
                    get_weights_accessor(data_path, "gconv3_" + unit_id_name + "_w_0.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    dwc_info)
                .set_name(unit_name + "/gconv3_" + unit_id_name + "/depthwise")
                << BatchNormalizationLayer(
                    get_weights_accessor(data_path, "gconv3_" + unit_id_name + "_bn_rm_0.npy"),
                    get_weights_accessor(data_path, "gconv3_" + unit_id_name + "_bn_riv_0.npy"),
                    get_weights_accessor(data_path, "gconv3_" + unit_id_name + "_bn_s_0.npy"),
                    get_weights_accessor(data_path, "gconv3_" + unit_id_name + "_bn_b_0.npy"),
                    1e-5f)
                .set_name(unit_name + "/gconv3_" + unit_id_name + "/BatchNorm")
                << ConvolutionLayer(
                    1U, 1U, depth,
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_1_name + "_w_0.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(1, 1, 0, 0), num_groups)
                .set_name(unit_name + "/gconv1_" + gconv_id_1_name + "/convolution")
                << BatchNormalizationLayer(
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_1_name + "_bn_rm_0.npy"),
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_1_name + "_bn_riv_0.npy"),
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_1_name + "_bn_s_0.npy"),
                    get_weights_accessor(data_path, "gconv1_" + gconv_id_1_name + "_bn_b_0.npy"),
                    1e-5f)
                .set_name(unit_name + "/gconv1_" + gconv_id_1_name + "/BatchNorm");

        if(stride == 2)
        {
            graph << ConcatLayer(std::move(left_ss), std::move(right_ss)).set_name(unit_name + "/Concat");
        }
        else
        {
            graph << EltwiseLayer(std::move(left_ss), std::move(right_ss), EltwiseOperation::Add).set_name(unit_name + "/Add");
        }
        graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "/Relu");
    }
};

/** Main program for ShuffleNet
 *
 * Model is based on:
 *      https://arxiv.org/abs/1707.01083
 *      "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"
 *      Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
 *
 * Provenance: https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<ShuffleNetExample>(argc, argv);
}
