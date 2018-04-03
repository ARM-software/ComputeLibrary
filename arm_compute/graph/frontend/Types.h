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
#ifndef __ARM_COMPUTE_GRAPH_STREAM_TYPES_H__
#define __ARM_COMPUTE_GRAPH_STREAM_TYPES_H__

#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
// Import types for graph
using graph::DataType;
using graph::TensorShape;

using graph::ActivationLayerInfo;
using graph::NormalizationLayerInfo;
using graph::NormType;
using graph::PadStrideInfo;
using graph::PoolingLayerInfo;
using graph::PoolingType;
using graph::Target;
using graph::ConvolutionMethod;
using graph::DepthwiseConvolutionMethod;
using graph::TensorDescriptor;
using graph::DimensionRoundingType;
using graph::GraphConfig;

/** Branch layer merging method */
enum class BranchMergeMethod
{
    DEPTH_CONCATENATE, /**< Concatenate across depth */
    ADD                /**< Adds the results of each stream */
};

/** Hints that can be passed to the stream to expose parameterization */
struct StreamHints
{
    Target                     target_hint                       = { Target::UNSPECIFIED };                 /**< Target execution hint */
    ConvolutionMethod          convolution_method_hint           = { ConvolutionMethod::DEFAULT };          /**< Convolution method hint */
    DepthwiseConvolutionMethod depthwise_convolution_method_hint = { DepthwiseConvolutionMethod::DEFAULT }; /**< Depthwise Convolution method hint */
};
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_STREAM_TYPES_H__ */