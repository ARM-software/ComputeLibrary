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
#include "arm_compute/graph2.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <tuple>

using namespace arm_compute::utils;
using namespace arm_compute::graph2::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement InceptionV4's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
class InceptionV4Example final : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Set target. 0 (NEON), 1 (OpenCL). By default it is NEON
        const int target                   = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        Target    target_hint              = set_target_hint2(target);
        bool      enable_tuning            = (target == 2);
        bool      enable_memory_management = true;

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

        graph << target_hint << InputLayer(TensorDescriptor(TensorShape(299U, 299U, 3U, 1U), DataType::F32),
                                           get_input_accessor(image, std::move(preprocessor), false))

              // Conv2d_1a_3x3
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_weights.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Conv2d_2a_3x3
              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_weights.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2a_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Conv2d_2b_3x3
              << ConvolutionLayer(3U, 3U, 64U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_weights.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_moving_mean.npy"),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_moving_variance.npy"),
                                         get_random_accessor(1.f, 1.f),
                                         get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Conv2d_2b_3x3_BatchNorm_beta.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

              << get_mixed_3a(data_path)
              << get_mixed_4a(data_path)
              << get_mixed_5a(data_path)
              // 4 inception A blocks
              << get_inceptionA_block(data_path, "Mixed_5b")
              << get_inceptionA_block(data_path, "Mixed_5c")
              << get_inceptionA_block(data_path, "Mixed_5d")
              << get_inceptionA_block(data_path, "Mixed_5e")
              // reduction A block
              << get_reductionA_block(data_path)
              // 7 inception B blocks
              << get_inceptionB_block(data_path, "Mixed_6b")
              << get_inceptionB_block(data_path, "Mixed_6c")
              << get_inceptionB_block(data_path, "Mixed_6d")
              << get_inceptionB_block(data_path, "Mixed_6e")
              << get_inceptionB_block(data_path, "Mixed_6f")
              << get_inceptionB_block(data_path, "Mixed_6g")
              << get_inceptionB_block(data_path, "Mixed_6h")
              // reduction B block
              << get_reductionB_block(data_path)
              // 3 inception C blocks
              << get_inceptionC_block(data_path, "Mixed_7b")
              << get_inceptionC_block(data_path, "Mixed_7c")
              << get_inceptionC_block(data_path, "Mixed_7d")
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << FlattenLayer()
              << FullyConnectedLayer(
                  1001U,
                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Logits_Logits_weights.npy"),
                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/Logits_Logits_biases.npy"))
              << SoftmaxLayer()
              << OutputLayer(get_output_accessor(label, 5));

        // Finalize graph
        graph.finalize(target_hint, enable_tuning, enable_memory_management);
    }

    void do_run() override
    {
        graph.run();
    }

private:
    Stream graph{ 0, "InceptionV4" };

private:
    BranchLayer get_mixed_3a(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_3a_";

        SubStream i_a(graph);
        i_a << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }

    BranchLayer get_mixed_4a(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_4a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }

    BranchLayer get_mixed_5a(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_5a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(3U, 3U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }

    BranchLayer get_inceptionA_block(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 64U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(1U, 1U, 96U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    BranchLayer get_reductionA_block(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_6a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(3U, 3U, 384U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c));
    }

    BranchLayer get_inceptionB_block(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 384U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 224U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(1U, 1U, 128U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    BranchLayer get_reductionB_block(const std::string &data_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/Mixed_7a_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 192U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(7U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x7_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(1U, 7U, 320U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_7x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(3U, 3U, 320U,
                                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_1a_3x3_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL), true));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c));
    }

    BranchLayer get_inceptionC_block(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";

        SubStream i_a(graph);
        i_a << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_0_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b1(static_cast<IStream &>(i_b));
        i_b1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0b_1x3_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b2(static_cast<IStream &>(i_b));
        i_b2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_1_Conv2d_0c_3x1_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // Merge b1 and b2
        i_b << BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_b1), std::move(i_b2));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0a_1x1_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                1U, 3U, 448U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 1))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0b_3x1_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 1U, 512U,
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_weights.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_mean.npy"),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_moving_variance.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0c_1x3_BatchNorm_beta.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c1(static_cast<IStream &>(i_c));
        i_c1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0d_1x3_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c2(static_cast<IStream &>(i_c));
        i_c2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_weights.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_moving_mean.npy"),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_moving_variance.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "Branch_2_Conv2d_0e_3x1_BatchNorm_beta.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // Merge i_c1 and i_c2
        i_c << BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_c1), std::move(i_c2));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(1U, 1U, 256U,
                                get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_weights.npy"),
                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_mean.npy"),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_moving_variance.npy"),
                                       get_random_accessor(1.f, 1.f),
                                       get_weights_accessor(data_path, total_path + "Branch_3_Conv2d_0b_1x1_BatchNorm_beta.npy"),
                                       0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Inception V4
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<InceptionV4Example>(argc, argv);
}
