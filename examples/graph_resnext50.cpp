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

/** Example demonstrating how to implement ResNeXt50 network using the Compute Library's graph API */
class GraphResNeXt50Example : public Example
{
public:
    GraphResNeXt50Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "ResNeXt50")
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
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params))
              << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/resnext50_model/bn_data_mul.npy"),
                            get_weights_accessor(data_path, "/cnn_data/resnext50_model/bn_data_add.npy"))
              .set_name("bn_data/Scale")
              << ConvolutionLayer(
                  7U, 7U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/resnext50_model/conv0_weights.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/resnext50_model/conv0_biases.npy"),
                  PadStrideInfo(2, 2, 2, 3, 2, 3, DimensionRoundingType::FLOOR))
              .set_name("conv0/Convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv0/Relu")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))).set_name("pool0");

        add_residual_block(data_path, weights_layout, /*ofm*/ 256, /*stage*/ 1, /*num_unit*/ 3, /*stride_conv_unit1*/ 1);
        add_residual_block(data_path, weights_layout, 512, 2, 4, 2);
        add_residual_block(data_path, weights_layout, 1024, 3, 6, 2);
        add_residual_block(data_path, weights_layout, 2048, 4, 3, 2);

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("pool1")
              << FlattenLayer().set_name("predictions/Reshape")
              << OutputLayer(get_npy_output_accessor(common_params.labels, TensorShape(2048U), DataType::F32));

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

    void add_residual_block(const std::string &data_path, DataLayout weights_layout,
                            unsigned int base_depth, unsigned int stage, unsigned int num_units, unsigned int stride_conv_unit1)
    {
        for(unsigned int i = 0; i < num_units; ++i)
        {
            std::stringstream unit_path_ss;
            unit_path_ss << "/cnn_data/resnext50_model/stage" << stage << "_unit" << (i + 1) << "_";
            std::string unit_path = unit_path_ss.str();

            std::stringstream unit_name_ss;
            unit_name_ss << "stage" << stage << "/unit" << (i + 1) << "/";
            std::string unit_name = unit_name_ss.str();

            PadStrideInfo pad_grouped_conv(1, 1, 1, 1);
            if(i == 0)
            {
                pad_grouped_conv = (stage == 1) ? PadStrideInfo(stride_conv_unit1, stride_conv_unit1, 1, 1) : PadStrideInfo(stride_conv_unit1, stride_conv_unit1, 0, 1, 0, 1, DimensionRoundingType::FLOOR);
            }

            SubStream right(graph);
            right << ConvolutionLayer(
                      1U, 1U, base_depth / 2,
                      get_weights_accessor(data_path, unit_path + "conv1_weights.npy", weights_layout),
                      get_weights_accessor(data_path, unit_path + "conv1_biases.npy"),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv1/convolution")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(
                      3U, 3U, base_depth / 2,
                      get_weights_accessor(data_path, unit_path + "conv2_weights.npy", weights_layout),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      pad_grouped_conv, 32)
                  .set_name(unit_name + "conv2/convolution")
                  << ScaleLayer(get_weights_accessor(data_path, unit_path + "bn2_mul.npy"),
                                get_weights_accessor(data_path, unit_path + "bn2_add.npy"))
                  .set_name(unit_name + "conv1/Scale")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv2/Relu")

                  << ConvolutionLayer(
                      1U, 1U, base_depth,
                      get_weights_accessor(data_path, unit_path + "conv3_weights.npy", weights_layout),
                      get_weights_accessor(data_path, unit_path + "conv3_biases.npy"),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv3/convolution");

            SubStream left(graph);
            if(i == 0)
            {
                left << ConvolutionLayer(
                         1U, 1U, base_depth,
                         get_weights_accessor(data_path, unit_path + "sc_weights.npy", weights_layout),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(stride_conv_unit1, stride_conv_unit1, 0, 0))
                     .set_name(unit_name + "sc/convolution")
                     << ScaleLayer(get_weights_accessor(data_path, unit_path + "sc_bn_mul.npy"),
                                   get_weights_accessor(data_path, unit_path + "sc_bn_add.npy"))
                     .set_name(unit_name + "sc/scale");
            }

            graph << EltwiseLayer(std::move(left), std::move(right), EltwiseOperation::Add).set_name(unit_name + "add");
            graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
        }
    }
};

/** Main program for ResNeXt50
 *
 * Model is based on:
 *      https://arxiv.org/abs/1611.05431
 *      "Aggregated Residual Transformations for Deep Neural Networks"
 *      Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphResNeXt50Example>(argc, argv);
}
