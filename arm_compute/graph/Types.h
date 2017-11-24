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
#ifndef __ARM_COMPUTE_GRAPH_TYPES_H__
#define __ARM_COMPUTE_GRAPH_TYPES_H__

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/SubTensorInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/logging/Macros.h"

/** Create a default core logger
 *
 * @note It will eventually create all default loggers in don't exist
 */
#define ARM_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER()                                  \
    do                                                                             \
    {                                                                              \
        if(arm_compute::logging::LoggerRegistry::get().logger("GRAPH") == nullptr) \
        {                                                                          \
            arm_compute::logging::LoggerRegistry::get().create_reserved_loggers(); \
        }                                                                          \
    } while(false)

#define ARM_COMPUTE_LOG_GRAPH(log_level, x)    \
    ARM_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER(); \
    ARM_COMPUTE_LOG_STREAM("GRAPH", log_level, x)

#define ARM_COMPUTE_LOG_GRAPH_INFO(x)          \
    ARM_COMPUTE_CREATE_DEFAULT_GRAPH_LOGGER(); \
    ARM_COMPUTE_LOG_STREAM("GRAPH", arm_compute::logging::LogLevel::INFO, x)

namespace arm_compute
{
namespace graph
{
using arm_compute::ActivationLayerInfo;
using arm_compute::Coordinates;
using arm_compute::DataType;
using arm_compute::DimensionRoundingType;
using arm_compute::ITensorInfo;
using arm_compute::NormType;
using arm_compute::NormalizationLayerInfo;
using arm_compute::PadStrideInfo;
using arm_compute::PoolingLayerInfo;
using arm_compute::PoolingType;
using arm_compute::SubTensorInfo;
using arm_compute::TensorInfo;
using arm_compute::TensorShape;
using arm_compute::WeightsInfo;

using arm_compute::logging::LogLevel;
using arm_compute::ConvertPolicy;

/**< Execution hint to the graph executor */
enum class TargetHint
{
    DONT_CARE, /**< Run node in any device */
    OPENCL,    /**< Run node on an OpenCL capable device (GPU) */
    NEON       /**< Run node on a NEON capable device */
};

/** Convolution method hint to the graph executor */
enum class ConvolutionMethodHint
{
    GEMM,  /**< Convolution using GEMM */
    DIRECT /**< Direct convolution */
};

/** Supported layer operations */
enum class OperationType
{
    ActivationLayer,
    BatchNormalizationLayer,
    ConvolutionLayer,
    DepthConvertLayer,
    DepthwiseConvolutionLayer,
    DequantizationLayer,
    FlattenLayer,
    FloorLayer,
    FullyConnectedLayer,
    L2NormalizeLayer,
    NormalizationLayer,
    PoolingLayer,
    QuantizationLayer,
    ReshapeLayer,
    SoftmaxLayer
};

/** Branch layer merging method */
enum class BranchMergeMethod
{
    DEPTH_CONCATENATE /**< Concatenate across depth */
};
} // namespace graph
} // namespace arm_compute
#endif /*__ARM_COMPUTE_GRAPH_TYPES_H__*/
