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
#include "arm_compute/graph/SubGraph.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/Scheduler.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <tuple>

using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

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

BranchLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, unsigned int expand1_filt, unsigned int expand3_filt)
{
    std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
    SubGraph    i_a;
    i_a << ConvolutionLayer(
            1U, 1U, expand1_filt,
            get_accessor(data_path, total_path + "expand1x1_w.npy"),
            get_accessor(data_path, total_path + "expand1x1_b.npy"),
            PadStrideInfo(1, 1, 0, 0))
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    SubGraph i_b;
    i_b << ConvolutionLayer(
            3U, 3U, expand3_filt,
            get_accessor(data_path, total_path + "expand3x3_w.npy"),
            get_accessor(data_path, total_path + "expand3x3_b.npy"),
            PadStrideInfo(1, 1, 1, 1))
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
}

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] batches )
 */
void main_graph_squeezenet(int argc, const char **argv)
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
    if(arm_compute::opencl_is_available())
    {
        arm_compute::CLScheduler::get().default_init();
    }

    Graph graph;
    arm_compute::Logger::get().set_logger(std::cout, arm_compute::LoggerVerbosity::INFO);

    graph << TargetHint::OPENCL
          << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, batches), 1, DataType::F32), DummyAccessor())
          << ConvolutionLayer(
              7U, 7U, 96U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
              PadStrideInfo(2, 2, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
          << ConvolutionLayer(
              1U, 1U, 16U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire2", 64U, 64U)
          << ConvolutionLayer(
              1U, 1U, 16U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire3", 64U, 64U)
          << ConvolutionLayer(
              1U, 1U, 32U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire4", 128U, 128U)
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
          << ConvolutionLayer(
              1U, 1U, 32U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire5", 128U, 128U)
          << ConvolutionLayer(
              1U, 1U, 48U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire6", 192U, 192U)
          << ConvolutionLayer(
              1U, 1U, 48U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire7", 192U, 192U)
          << ConvolutionLayer(
              1U, 1U, 64U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire8", 256U, 256U)
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
          << ConvolutionLayer(
              1U, 1U, 64U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << get_expand_fire_node(data_path, "fire9", 256U, 256U)
          << ConvolutionLayer(
              1U, 1U, 1000U,
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy"),
              get_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 13, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL)))
          << SoftmaxLayer()
          << Tensor(DummyAccessor());

    graph.run();
}

/** Main program for Squeezenet v1.0
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] batches )
 */
int main(int argc, const char **argv)
{
    return arm_compute::utils::run_example(argc, argv, main_graph_squeezenet);
}
