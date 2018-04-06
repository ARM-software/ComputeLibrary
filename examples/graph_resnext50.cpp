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
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement ResNeXt50 network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] npy_in, [optional] npy_out, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
class GraphResNeXt50Example : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string npy_in;    /* Input npy data */
        std::string npy_out;   /* Output npy data */

        // Set target. 0 (NEON), 1 (OpenCL), 2 (OpenCL with Tuner). By default it is NEON
        const int    target         = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        Target       target_hint    = set_target_hint(target);
        FastMathHint fast_math_hint = FastMathHint::DISABLED;

        // Parse arguments
        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [target] [path_to_data] [npy_in] [npy_out] [fast_math_hint]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 2)
        {
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " [path_to_data] [npy_in] [npy_out] [fast_math_hint]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 3)
        {
            data_path = argv[2];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " [npy_in] [npy_out] [fast_math_hint]\n\n";
            std::cout << "No input npy file provided: using random values\n\n";
        }
        else if(argc == 4)
        {
            data_path = argv[2];
            npy_in    = argv[3];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " [npy_out] [fast_math_hint]\n\n";
            std::cout << "No output npy file provided: skipping output accessor\n\n";
        }
        else if(argc == 5)
        {
            data_path = argv[2];
            npy_in    = argv[3];
            npy_out   = argv[4];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " [fast_math_hint]\n\n";
            std::cout << "No fast math info provided: disabling fast math\n\n";
        }
        else
        {
            data_path      = argv[2];
            npy_in         = argv[3];
            npy_out        = argv[4];
            fast_math_hint = (std::strtol(argv[5], nullptr, 1) == 0) ? FastMathHint::DISABLED : FastMathHint::ENABLED;
        }

        graph << target_hint
              << fast_math_hint
              << InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), DataType::F32),
                            get_input_accessor(npy_in))
              << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/resnext50_model/bn_data_mul.npy"),
                            get_weights_accessor(data_path, "/cnn_data/resnext50_model/bn_data_add.npy"))
              .set_name("bn_data/Scale")
              << ConvolutionLayer(
                  7U, 7U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/resnext50_model/conv0_weights.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnext50_model/conv0_biases.npy"),
                  PadStrideInfo(2, 2, 2, 3, 2, 3, DimensionRoundingType::FLOOR))
              .set_name("conv0/Convolution")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv0/Relu")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))).set_name("pool0");

        add_residual_block(data_path, /*ofm*/ 256, /*stage*/ 1, /*num_unit*/ 3, /*stride_conv_unit1*/ 1);
        add_residual_block(data_path, 512, 2, 4, 2);
        add_residual_block(data_path, 1024, 3, 6, 2);
        add_residual_block(data_path, 2048, 4, 3, 2);

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("pool1")
              << FlattenLayer().set_name("predictions/Reshape")
              << OutputLayer(get_npy_output_accessor(npy_out, TensorShape(2048U), DataType::F32));

        // Finalize graph
        GraphConfig config;
        config.use_tuner = (target == 2);
        graph.finalize(target_hint, config);
    }

    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    Stream graph{ 0, "ResNeXt50" };

    void add_residual_block(const std::string &data_path, unsigned int base_depth, unsigned int stage, unsigned int num_units, unsigned int stride_conv_unit1)
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
                      get_weights_accessor(data_path, unit_path + "conv1_weights.npy"),
                      get_weights_accessor(data_path, unit_path + "conv1_biases.npy"),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv1/convolution")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(
                      3U, 3U, base_depth / 2,
                      get_weights_accessor(data_path, unit_path + "conv2_weights.npy"),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      pad_grouped_conv, 32)
                  .set_name(unit_name + "conv2/convolution")
                  << ScaleLayer(get_weights_accessor(data_path, unit_path + "bn2_mul.npy"),
                                get_weights_accessor(data_path, unit_path + "bn2_add.npy"))
                  .set_name(unit_name + "conv1/Scale")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv2/Relu")

                  << ConvolutionLayer(
                      1U, 1U, base_depth,
                      get_weights_accessor(data_path, unit_path + "conv3_weights.npy"),
                      get_weights_accessor(data_path, unit_path + "conv3_biases.npy"),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv3/convolution");

            SubStream left(graph);
            if(i == 0)
            {
                left << ConvolutionLayer(
                         1U, 1U, base_depth,
                         get_weights_accessor(data_path, unit_path + "sc_weights.npy"),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(stride_conv_unit1, stride_conv_unit1, 0, 0))
                     .set_name(unit_name + "sc/convolution")
                     << ScaleLayer(get_weights_accessor(data_path, unit_path + "sc_bn_mul.npy"),
                                   get_weights_accessor(data_path, unit_path + "sc_bn_add.npy"))
                     .set_name(unit_name + "sc/scale");
            }

            graph << BranchLayer(BranchMergeMethod::ADD, std::move(left), std::move(right)).set_name(unit_name + "add");
            graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
        }
    }
};

/** Main program for ResNeXt50
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [[optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] npy_in, [optional] npy_out )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphResNeXt50Example>(argc, argv);
}
