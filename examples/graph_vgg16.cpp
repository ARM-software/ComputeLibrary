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
#include "arm_compute/graph2.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute::utils;
using namespace arm_compute::graph2::frontend;
using namespace arm_compute::graph_utils;

namespace
{
/** This function checks if we can use GEMM-based convolution trying to allocate a memory of size "size_in_bytes"
 *
 * @param[in] size_in_bytes Memory size in bytes needed for VGG-16
 *
 * @return The convolution layer hint
 */
ConvolutionMethod convolution_hint_vgg16(size_t size_in_bytes)
{
    return ((get_mem_free_from_meminfo() * 1024) >= size_in_bytes) ? ConvolutionMethod::GEMM : ConvolutionMethod::DIRECT;
}
} // namespace

/** Example demonstrating how to implement VGG16's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
class GraphVGG16Example : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 123.68f, 116.779f, 103.939f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Set target. 0 (NEON), 1 (OpenCL), 2 (OpenCL with Tuner). By default it is NEON
        const int target                   = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        Target    target_hint              = set_target_hint2(target);
        bool      enable_tuning            = (target == 2);
        bool      enable_memory_management = true;

        // Check if we can use GEMM-based convolutions evaluating if the platform has at least 1.8 GB of available memory
        const size_t      memory_required           = 1932735283L;
        const bool        is_opencl                 = target_hint == Target::CL;
        ConvolutionMethod first_convolution3x3_hint = is_opencl ? ConvolutionMethod::DIRECT : ConvolutionMethod::GEMM;
        ConvolutionMethod convolution3x3_hint       = is_opencl ? ConvolutionMethod::WINOGRAD : convolution_hint_vgg16(memory_required);

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
              << first_convolution3x3_hint
              << InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), DataType::F32),
                            get_input_accessor(image, std::move(preprocessor)))
              // Layer 1
              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv1_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv1_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << convolution3x3_hint
              // Layer 2
              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv1_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv1_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 3
              << ConvolutionLayer(
                  3U, 3U, 128U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv2_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv2_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 4
              << ConvolutionLayer(
                  3U, 3U, 128U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv2_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv2_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 5
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv3_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv3_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 6
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv3_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv3_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 7
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv3_3_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv3_3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 8
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv4_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv4_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 9
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv4_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv4_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 10
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv4_3_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv4_3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 11
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv5_1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv5_1_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 12
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv5_2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv5_2_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 13
              << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv5_3_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/conv5_3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
              // Layer 14
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/fc6_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/fc6_b.npy"))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 15
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/fc7_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/fc7_b.npy"))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              // Layer 16
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/fc8_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/vgg16_model/fc8_b.npy"))
              // Softmax
              << SoftmaxLayer()
              << OutputLayer(get_output_accessor(label, 5));

        // Finalize graph
        graph.finalize(target_hint, enable_tuning, enable_memory_management);
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    Stream graph{ 0, "VGG16" };
};

/** Main program for VGG16
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphVGG16Example>(argc, argv);
}
