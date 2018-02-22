/*
 * Copyright (c) 2017-2018 ARM Limited.
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

/** Example demonstrating how to implement Microsoft's ResNet50 network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
class GraphResNet50Example : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb,
                                                                                                                   false /* Do not convert to BGR */);

        // Set target. 0 (NEON), 1 (OpenCL), 2 (OpenCL with Tuner). By default it is NEON
        const int  int_target_hint = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        TargetHint target_hint     = set_target_hint(int_target_hint);

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
                        get_input_accessor(image, std::move(preprocessor), false /* Do not convert to BGR */))
              << ConvolutionLayer(
                  7U, 7U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_weights.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 3, 3))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_beta.npy"),
                  0.0000100099996416f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)));

        add_residual_block(data_path, "block1", 64, 3, 2);
        add_residual_block(data_path, "block2", 128, 4, 2);
        add_residual_block(data_path, "block3", 256, 6, 2);
        add_residual_block(data_path, "block4", 512, 3, 1);

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << ConvolutionLayer(
                  1U, 1U, 1000U,
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_weights.npy"),
                  get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_biases.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << FlattenLayer()
              << SoftmaxLayer()
              << Tensor(get_output_accessor(label, 5));

        // In order to enable the OpenCL tuner, graph_init() has to be called only when all nodes have been instantiated
        graph.graph_init(int_target_hint == 2);
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    Graph graph{};

    void add_residual_block(const std::string &data_path, const std::string &name, unsigned int base_depth, unsigned int num_units, unsigned int stride)
    {
        for(unsigned int i = 0; i < num_units; ++i)
        {
            std::stringstream unit;
            unit << "/cnn_data/resnet50_model/" << name << "_unit_" << (i + 1) << "_bottleneck_v1_";
            std::string unit_name = unit.str();

            unsigned int middle_stride = 1;

            if(i == (num_units - 1))
            {
                middle_stride = stride;
            }

            SubGraph right;
            right << ConvolutionLayer(
                      1U, 1U, base_depth,
                      get_weights_accessor(data_path, unit_name + "conv1_weights.npy"),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(1, 1, 0, 0))
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_name + "conv1_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_name + "conv1_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_name + "conv1_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_name + "conv1_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

                  << ConvolutionLayer(
                      3U, 3U, base_depth,
                      get_weights_accessor(data_path, unit_name + "conv2_weights.npy"),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(middle_stride, middle_stride, 1, 1))
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_name + "conv2_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_name + "conv2_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_name + "conv2_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_name + "conv2_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

                  << ConvolutionLayer(
                      1U, 1U, base_depth * 4,
                      get_weights_accessor(data_path, unit_name + "conv3_weights.npy"),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(1, 1, 0, 0))
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_name + "conv3_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_name + "conv3_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_name + "conv3_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_name + "conv3_BatchNorm_beta.npy"),
                      0.0000100099996416f);

            if(i == 0)
            {
                SubGraph left;
                left << ConvolutionLayer(
                         1U, 1U, base_depth * 4,
                         get_weights_accessor(data_path, unit_name + "shortcut_weights.npy"),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(1, 1, 0, 0))
                     << BatchNormalizationLayer(
                         get_weights_accessor(data_path, unit_name + "shortcut_BatchNorm_moving_mean.npy"),
                         get_weights_accessor(data_path, unit_name + "shortcut_BatchNorm_moving_variance.npy"),
                         get_weights_accessor(data_path, unit_name + "shortcut_BatchNorm_gamma.npy"),
                         get_weights_accessor(data_path, unit_name + "shortcut_BatchNorm_beta.npy"),
                         0.0000100099996416f);

                graph << ResidualLayer(std::move(left), std::move(right));
            }
            else if(middle_stride > 1)
            {
                SubGraph left;
                left << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 1, PadStrideInfo(middle_stride, middle_stride, 0, 0), true))
                     << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 1.f, 0.f));

                graph << ResidualLayer(std::move(left), std::move(right));
            }
            else
            {
                graph << ResidualLayer(std::move(right));
            }

            graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        }
    }
};

/** Main program for ResNet50
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphResNet50Example>(argc, argv);
}
