/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute::utils;
using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement MobileNet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
class GraphMobilenetExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        constexpr float mean_r = 122.68f; /* Mean value to subtract from red channel */
        constexpr float mean_g = 116.67f; /* Mean value to subtract from green channel */
        constexpr float mean_b = 104.01f; /* Mean value to subtract from blue channel */

        // Set target. 0 (NEON), 1 (OpenCL). By default it is NEON
        TargetHint            target_hint      = set_target_hint(argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0);
        ConvolutionMethodHint convolution_hint = target_hint == TargetHint::NEON ? ConvolutionMethodHint::GEMM : ConvolutionMethodHint::DIRECT;

        // Parse arguments
        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [target] [path_to_data] [image] [labels]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 2)
        {
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " [path_to_data] [image] [labels]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 3)
        {
            data_path = argv[2];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " [image] [labels]\n\n";
            std::cout << "No image provided: using random values\n\n";
        }
        else if(argc == 4)
        {
            data_path = argv[2];
            image     = argv[3];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " [labels]\n\n";
            std::cout << "No text file with labels provided: skipping output accessor\n\n";
        }
        else
        {
            data_path = argv[2];
            image     = argv[3];
            label     = argv[4];
        }

        graph << target_hint
              << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, 1U), 1, DataType::F32),
                        get_input_accessor(image, mean_r, mean_g, mean_b))
              << convolution_hint
              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_v1_model/Conv2d_0_weights.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_v1_model/Conv2d_0_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_v1_model/Conv2d_0_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_v1_model/Conv2d_0_BatchNorm_beta.npy"),
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_v1_model/Conv2d_0_BatchNorm_gamma.npy"),
                  0.001f)

              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
              << get_dwsc_node(data_path, "Conv2d_1", 64, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_2", 128, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_3", 128, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_4", 256, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_5", 256, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_6", 512, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_7", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_8", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_9", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_10", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_11", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_12", 1024, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << get_dwsc_node(data_path, "Conv2d_13", 1024, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << ConvolutionLayer(
                  1U, 1U, 1001U,
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_v1_model/Logits_Conv2d_1c_1x1_weights.npy"),
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_v1_model/Logits_Conv2d_1c_1x1_biases.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ReshapeLayer(TensorShape(1001U))
              << SoftmaxLayer()
              << Tensor(get_output_accessor(label, 5));
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    Graph graph{};

    BranchLayer get_dwsc_node(const std::string &data_path, std::string &&param_path,
                              unsigned int  conv_filt,
                              PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
    {
        std::string total_path = "/cnn_data/mobilenet_v1_model/" + param_path + "_";
        SubGraph    sg;
        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy"),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               dwc_pad_stride_info,
               true)
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_beta.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_gamma.npy"),
               0.001f)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
           << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "pointwise_weights.npy"),
               std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
               conv_pad_stride_info)
           << BatchNormalizationLayer(
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_mean.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_variance.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_beta.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_gamma.npy"),
               0.001f)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));

        return BranchLayer(std::move(sg));
    }
};

/** Main program for MobileNetV1
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphMobilenetExample>(argc, argv);
}
