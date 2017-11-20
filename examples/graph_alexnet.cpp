/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Logger.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/Scheduler.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>

using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

/** Generates appropriate accessor according to the specified path
 *
 * @note If path is empty will generate a DummyAccessor else will generate a NumPyBinLoader
 *
 * @param[in] path      Path to the data files
 * @param[in] data_file Relative path to the data files from path
 *
 * @return An appropriate tensor accessor
 */
std::unique_ptr<ITensorAccessor> get_accessor(const std::string &path, const std::string &data_file)
{
    if(path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<NumPyBinLoader>(path + data_file);
    }
}

/** Generates appropriate input accessor according to the specified ppm_path
 *
 * @note If ppm_path is empty will generate a DummyAccessor else will generate a PPMAccessor
 *
 * @param[in] ppm_path Path to PPM file
 * @param[in] mean_r   Red mean value to be subtracted from red channel
 * @param[in] mean_g   Green mean value to be subtracted from green channel
 * @param[in] mean_b   Blue mean value to be subtracted from blue channel
 *
 * @return An appropriate tensor accessor
 */
std::unique_ptr<ITensorAccessor> get_input_accessor(const std::string &ppm_path, float mean_r, float mean_g, float mean_b)
{
    if(ppm_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<PPMAccessor>(ppm_path, true, mean_r, mean_g, mean_b);
    }
}

/** Generates appropriate output accessor according to the specified labels_path
 *
 * @note If labels_path is empty will generate a DummyAccessor else will generate a TopNPredictionsAccessor
 *
 * @param[in]  labels_path   Path to labels text file
 * @param[in]  top_n         (Optional) Number of output classes to print
 * @param[out] output_stream (Optional) Output stream
 *
 * @return An appropriate tensor accessor
 */
std::unique_ptr<ITensorAccessor> get_output_accessor(const std::string &labels_path, size_t top_n = 5, std::ostream &output_stream = std::cout)
{
    if(labels_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<TopNPredictionsAccessor>(labels_path, top_n, output_stream);
    }
}

/** Example demonstrating how to implement AlexNet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
void main_graph_alexnet(int argc, const char **argv)
{
    std::string data_path; /* Path to the trainable data */
    std::string image;     /* Image data */
    std::string label;     /* Label data */

    constexpr float mean_r = 122.68f; /* Mean value to subtract from red channel */
    constexpr float mean_g = 116.67f; /* Mean value to subtract from green channel */
    constexpr float mean_b = 104.01f; /* Mean value to subtract from blue channel */

    // Parse arguments
    if(argc < 2)
    {
        // Print help
        std::cout << "Usage: " << argv[0] << " [path_to_data] [image] [labels]\n\n";
        std::cout << "No data folder provided: using random values\n\n";
    }
    else if(argc == 2)
    {
        data_path = argv[1];
        std::cout << "Usage: " << argv[0] << " " << argv[1] << " [image] [labels]\n\n";
        std::cout << "No image provided: using random values\n\n";
    }
    else if(argc == 3)
    {
        data_path = argv[1];
        image     = argv[2];
        std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " [labels]\n\n";
        std::cout << "No text file with labels provided: skipping output accessor\n\n";
    }
    else
    {
        data_path = argv[1];
        image     = argv[2];
        label     = argv[3];
    }

    // Check if OpenCL is available and initialize the scheduler
    TargetHint hint = TargetHint::NEON;
    if(arm_compute::opencl_is_available())
    {
        arm_compute::CLScheduler::get().default_init();
        hint = TargetHint::OPENCL;
    }

    Graph graph;
    arm_compute::Logger::get().set_logger(std::cout, arm_compute::LoggerVerbosity::INFO);

    graph << hint
          << Tensor(TensorInfo(TensorShape(227U, 227U, 3U, 1U), 1, DataType::F32),
                    get_input_accessor(image, mean_r, mean_g, mean_b))
          // Layer 1
          << ConvolutionLayer(
              11U, 11U, 96U,
              get_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
              PadStrideInfo(4, 4, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)))
          // Layer 2
          << ConvolutionMethodHint::DIRECT
          << ConvolutionLayer(
              5U, 5U, 256U,
              get_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"),
              PadStrideInfo(1, 1, 2, 2), 2)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)))
          // Layer 3
          << ConvolutionLayer(
              3U, 3U, 384U,
              get_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
              PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 4
          << ConvolutionLayer(
              3U, 3U, 384U,
              get_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
              PadStrideInfo(1, 1, 1, 1), 2)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 5
          << ConvolutionLayer(
              3U, 3U, 256U,
              get_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
              PadStrideInfo(1, 1, 1, 1), 2)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)))
          // Layer 6
          << FullyConnectedLayer(
              4096U,
              get_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 7
          << FullyConnectedLayer(
              4096U,
              get_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 8
          << FullyConnectedLayer(
              1000U,
              get_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy"),
              get_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
          // Softmax
          << SoftmaxLayer()
          << Tensor(get_output_accessor(label, 5));

    // Run graph
    graph.run();
}

/** Main program for AlexNet
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, const char **argv)
{
    return arm_compute::utils::run_example(argc, argv, main_graph_alexnet);
}
