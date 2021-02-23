/*
 * Copyright (c) 2019-2021 Arm Limited.
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

/** Example demonstrating how to implement Mnist's network using the Compute Library's graph API */
class GraphMnistExample : public Example
{
public:
    GraphMnistExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "LeNet")
    {
    }
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

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty() && arm_compute::is_data_type_quantized_asymmetric(common_params.data_type))
        {
            data_path += "/cnn_data/mnist_qasymm8_model/";
        }

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape     = permute_shape(TensorShape(28U, 28U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        const QuantizationInfo in_quant_info = QuantizationInfo(0.003921568859368563f, 0);

        const std::vector<std::pair<QuantizationInfo, QuantizationInfo>> conv_quant_info =
        {
            { QuantizationInfo(0.004083447158336639f, 138), QuantizationInfo(0.0046257381327450275f, 0) }, // conv0
            { QuantizationInfo(0.0048590428195893764f, 149), QuantizationInfo(0.03558270260691643f, 0) },  // conv1
            { QuantizationInfo(0.004008443560451269f, 146), QuantizationInfo(0.09117382764816284f, 0) },   // conv2
            { QuantizationInfo(0.004344311077147722f, 160), QuantizationInfo(0.5494495034217834f, 167) },  // fc
        };

        // Set weights trained layout
        const DataLayout        weights_layout = DataLayout::NHWC;
        FullyConnectedLayerInfo fc_info        = FullyConnectedLayerInfo();
        fc_info.set_weights_trained_layout(weights_layout);

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor.set_quantization_info(in_quant_info),
                            get_input_accessor(common_params))
              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "conv2d_weights_quant_FakeQuantWithMinMaxVars.npy", weights_layout),
                  get_weights_accessor(data_path, "conv2d_Conv2D_bias.npy"),
                  PadStrideInfo(1U, 1U, 1U, 1U), 1, conv_quant_info.at(0).first, conv_quant_info.at(0).second)
              .set_name("Conv0")

              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "conv2d_1_weights_quant_FakeQuantWithMinMaxVars.npy", weights_layout),
                  get_weights_accessor(data_path, "conv2d_1_Conv2D_bias.npy"),
                  PadStrideInfo(1U, 1U, 1U, 1U), 1, conv_quant_info.at(1).first, conv_quant_info.at(1).second)
              .set_name("conv1")

              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("maxpool1")

              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "conv2d_2_weights_quant_FakeQuantWithMinMaxVars.npy", weights_layout),
                  get_weights_accessor(data_path, "conv2d_2_Conv2D_bias.npy"),
                  PadStrideInfo(1U, 1U, 1U, 1U), 1, conv_quant_info.at(2).first, conv_quant_info.at(2).second)
              .set_name("conv2")

              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("maxpool2")

              << FullyConnectedLayer(
                  10U,
                  get_weights_accessor(data_path, "dense_weights_quant_FakeQuantWithMinMaxVars_transpose.npy", weights_layout),
                  get_weights_accessor(data_path, "dense_MatMul_bias.npy"),
                  fc_info, conv_quant_info.at(3).first, conv_quant_info.at(3).second)
              .set_name("fc")

              << SoftmaxLayer().set_name("prob");

        if(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type))
        {
            graph << DequantizationLayer().set_name("dequantize");
        }

        graph << OutputLayer(get_output_accessor(common_params, 5));

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
        // Run graph
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
};

/** Main program for Mnist Example
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphMnistExample>(argc, argv);
}
