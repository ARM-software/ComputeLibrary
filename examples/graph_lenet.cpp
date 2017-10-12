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

#include "arm_compute/core/utils/logging/LoggerRegistry.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/Scheduler.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>

using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;
using namespace arm_compute::logging;

/** Generates appropriate accessor according to the specified path
 *
 * @note If path is empty will generate a DummyAccessor else will generate a NumPyBinLoader
 *
 * @param path       Path to the data files
 * @param data_file  Relative path to the data files from path
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

/** Example demonstrating how to implement LeNet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] batches )
 */
void main_graph_lenet(int argc, const char **argv)
{
    std::string  data_path;   /** Path to the trainable data */
    unsigned int batches = 4; /** Number of batches */

    // Parse arguments
    if(argc < 2)
    {
        // Print help
        std::cout << "Usage: " << argv[0] << " [path_to_data] [batches]\n\n";
        std::cout << "No data folder provided: using random values\n\n";
    }
    else if(argc == 2)
    {
        //Do something with argv[1]
        data_path = argv[1];
        std::cout << "Usage: " << argv[0] << " [path_to_data] [batches]\n\n";
        std::cout << "No number of batches where specified, thus will use the default : " << batches << "\n\n";
    }
    else
    {
        //Do something with argv[1] and argv[2]
        data_path = argv[1];
        batches   = std::strtol(argv[2], nullptr, 0);
    }

    // Check if OpenCL is available and initialize the scheduler
    TargetHint hint = TargetHint::NEON;
    if(arm_compute::opencl_is_available())
    {
        arm_compute::CLScheduler::get().default_init();
        hint = TargetHint::OPENCL;
    }

    Graph graph;
    LoggerRegistry::get().create_reserved_loggers(LogLevel::INFO, { std::make_shared<StdPrinter>() });

    //conv1 << pool1 << conv2 << pool2 << fc1 << act1 << fc2 << smx
    graph << hint
          << Tensor(TensorInfo(TensorShape(28U, 28U, 1U, batches), 1, DataType::F32), DummyAccessor())
          << ConvolutionLayer(
              5U, 5U, 20U,
              get_accessor(data_path, "/cnn_data/lenet_model/conv1_w.npy"),
              get_accessor(data_path, "/cnn_data/lenet_model/conv1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          << ConvolutionLayer(
              5U, 5U, 50U,
              get_accessor(data_path, "/cnn_data/lenet_model/conv2_w.npy"),
              get_accessor(data_path, "/cnn_data/lenet_model/conv2_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          << FullyConnectedLayer(
              500U,
              get_accessor(data_path, "/cnn_data/lenet_model/ip1_w.npy"),
              get_accessor(data_path, "/cnn_data/lenet_model/ip1_b.npy"))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << FullyConnectedLayer(
              10U,
              get_accessor(data_path, "/cnn_data/lenet_model/ip2_w.npy"),
              get_accessor(data_path, "/cnn_data/lenet_model/ip2_b.npy"))
          << SoftmaxLayer()
          << Tensor(DummyAccessor());

    graph.run();
}

/** Main program for LeNet
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] batches )
 */
int main(int argc, const char **argv)
{
    return arm_compute::utils::run_example(argc, argv, main_graph_lenet);
}
