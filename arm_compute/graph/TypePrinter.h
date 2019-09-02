/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_TYPE_PRINTER_H__
#define __ARM_COMPUTE_GRAPH_TYPE_PRINTER_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/graph/Types.h"

#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace graph
{
/** Formatted output of the Target. */
inline ::std::ostream &operator<<(::std::ostream &os, const Target &target)
{
    switch(target)
    {
        case Target::UNSPECIFIED:
            os << "UNSPECIFIED";
            break;
        case Target::NEON:
            os << "NEON";
            break;
        case Target::CL:
            os << "CL";
            break;
        case Target::GC:
            os << "GC";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline ::std::ostream &operator<<(::std::ostream &os, const NodeType &node_type)
{
    switch(node_type)
    {
        case NodeType::ActivationLayer:
            os << "ActivationLayer";
            break;
        case NodeType::BatchNormalizationLayer:
            os << "BatchNormalizationLayer";
            break;
        case NodeType::BoundingBoxTransformLayer:
            os << "BoundingBoxTransformLayer";
            break;
        case NodeType::ChannelShuffleLayer:
            os << "ChannelShuffleLayer";
            break;
        case NodeType::ConcatenateLayer:
            os << "ConcatenateLayer";
            break;
        case NodeType::ConvolutionLayer:
            os << "ConvolutionLayer";
            break;
        case NodeType::DeconvolutionLayer:
            os << "DeconvolutionLayer";
            break;
        case NodeType::DetectionOutputLayer:
            os << "DetectionOutputLayer";
            break;
        case NodeType::DetectionPostProcessLayer:
            os << "DetectionPostProcessLayer";
            break;
        case NodeType::DepthwiseConvolutionLayer:
            os << "DepthwiseConvolutionLayer";
            break;
        case NodeType::EltwiseLayer:
            os << "EltwiseLayer";
            break;
        case NodeType::FlattenLayer:
            os << "FlattenLayer";
            break;
        case NodeType::FullyConnectedLayer:
            os << "FullyConnectedLayer";
            break;
        case NodeType::FusedConvolutionBatchNormalizationLayer:
            os << "FusedConvolutionBatchNormalizationLayer";
            break;
        case NodeType::FusedDepthwiseConvolutionBatchNormalizationLayer:
            os << "FusedDepthwiseConvolutionBatchNormalizationLayer";
            break;
        case NodeType::GenerateProposalsLayer:
            os << "GenerateProposalsLayer";
            break;
        case NodeType::NormalizationLayer:
            os << "NormalizationLayer";
            break;
        case NodeType::NormalizePlanarYUVLayer:
            os << "NormalizePlanarYUVLayer";
            break;
        case NodeType::PadLayer:
            os << "PadLayer";
            break;
        case NodeType::PermuteLayer:
            os << "PermuteLayer";
            break;
        case NodeType::PoolingLayer:
            os << "PoolingLayer";
            break;
        case NodeType::PriorBoxLayer:
            os << "PriorBoxLayer";
            break;
        case NodeType::QuantizationLayer:
            os << "QuantizationLayer";
            break;
        case NodeType::ReorgLayer:
            os << "ReorgLayer";
            break;
        case NodeType::ReshapeLayer:
            os << "ReshapeLayer";
            break;
        case NodeType::ResizeLayer:
            os << "ResizeLayer";
            break;
        case NodeType::ROIAlignLayer:
            os << "ROIAlignLayer";
            break;
        case NodeType::SoftmaxLayer:
            os << "SoftmaxLayer";
            break;
        case NodeType::SliceLayer:
            os << "SliceLayer";
            break;
        case NodeType::SplitLayer:
            os << "SplitLayer";
            break;
        case NodeType::StackLayer:
            os << "StackLayer";
            break;
        case NodeType::UpsampleLayer:
            os << "UpsampleLayer";
            break;
        case NodeType::YOLOLayer:
            os << "YOLOLayer";
            break;
        case NodeType::Input:
            os << "Input";
            break;
        case NodeType::Output:
            os << "Output";
            break;
        case NodeType::Const:
            os << "Const";
            break;
        case NodeType::Dummy:
            os << "Dummy";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the EltwiseOperation type. */
inline ::std::ostream &operator<<(::std::ostream &os, const EltwiseOperation &eltwise_op)
{
    switch(eltwise_op)
    {
        case EltwiseOperation::Add:
            os << "Add";
            break;
        case EltwiseOperation::Mul:
            os << "Mul";
            break;
        case EltwiseOperation::Sub:
            os << "Sub";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the ConvolutionMethod type. */
inline ::std::ostream &operator<<(::std::ostream &os, const ConvolutionMethod &method)
{
    switch(method)
    {
        case ConvolutionMethod::Default:
            os << "Default";
            break;
        case ConvolutionMethod::Direct:
            os << "Direct";
            break;
        case ConvolutionMethod::GEMM:
            os << "GEMM";
            break;
        case ConvolutionMethod::Winograd:
            os << "Winograd";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the FastMathHint type. */
inline ::std::ostream &operator<<(::std::ostream &os, const FastMathHint &hint)
{
    switch(hint)
    {
        case FastMathHint::Enabled:
            os << "Enabled";
            break;
        case FastMathHint::Disabled:
            os << "Disabled";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the DepthwiseConvolutionMethod type. */
inline ::std::ostream &operator<<(::std::ostream &os, const DepthwiseConvolutionMethod &method)
{
    switch(method)
    {
        case DepthwiseConvolutionMethod::Default:
            os << "DEFAULT";
            break;
        case DepthwiseConvolutionMethod::GEMV:
            os << "GEMV";
            break;
        case DepthwiseConvolutionMethod::Optimized3x3:
            os << "Optimized3x3";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_TYPE_PRINTER_H__ */
