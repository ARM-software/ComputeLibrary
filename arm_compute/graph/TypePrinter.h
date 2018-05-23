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
#ifndef __ARM_COMPUTE_GRAPH_TYPE_PRINTER_H__
#define __ARM_COMPUTE_GRAPH_TYPE_PRINTER_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
/** Formatted output of the Dimensions type. */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const arm_compute::Dimensions<T> &dimensions)
{
    if(dimensions.num_dimensions() > 0)
    {
        os << dimensions[0];

        for(unsigned int d = 1; d < dimensions.num_dimensions(); ++d)
        {
            os << "x" << dimensions[d];
        }
    }

    return os;
}

/** Formatted output of the Size2D type. */
inline ::std::ostream &operator<<(::std::ostream &os, const Size2D &size)
{
    os << size.width << "x" << size.height;

    return os;
}

/** Formatted output of the DataType type. */
inline ::std::ostream &operator<<(::std::ostream &os, const DataType &data_type)
{
    switch(data_type)
    {
        case DataType::UNKNOWN:
            os << "UNKNOWN";
            break;
        case DataType::U8:
            os << "U8";
            break;
        case DataType::QS8:
            os << "QS8";
            break;
        case DataType::QASYMM8:
            os << "QASYMM8";
            break;
        case DataType::S8:
            os << "S8";
            break;
        case DataType::U16:
            os << "U16";
            break;
        case DataType::S16:
            os << "S16";
            break;
        case DataType::QS16:
            os << "QS16";
            break;
        case DataType::U32:
            os << "U32";
            break;
        case DataType::S32:
            os << "S32";
            break;
        case DataType::U64:
            os << "U64";
            break;
        case DataType::S64:
            os << "S64";
            break;
        case DataType::F16:
            os << "F16";
            break;
        case DataType::F32:
            os << "F32";
            break;
        case DataType::F64:
            os << "F64";
            break;
        case DataType::SIZET:
            os << "SIZET";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

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
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the DataLayout */
inline ::std::ostream &operator<<(::std::ostream &os, const DataLayout &data_layout)
{
    switch(data_layout)
    {
        case DataLayout::NCHW:
            os << "NCHW";
            break;
        case DataLayout::NHWC:
            os << "NHWC";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the activation function type. */
inline ::std::ostream &operator<<(::std::ostream &os, const ActivationLayerInfo::ActivationFunction &act_function)
{
    switch(act_function)
    {
        case ActivationLayerInfo::ActivationFunction::ABS:
            os << "ABS";
            break;
        case ActivationLayerInfo::ActivationFunction::LINEAR:
            os << "LINEAR";
            break;
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            os << "LOGISTIC";
            break;
        case ActivationLayerInfo::ActivationFunction::RELU:
            os << "RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
            os << "BOUNDED_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            os << "LEAKY_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            os << "SOFT_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::SQRT:
            os << "SQRT";
            break;
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            os << "LU_BOUNDED_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::SQUARE:
            os << "SQUARE";
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            os << "TANH";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const ActivationLayerInfo::ActivationFunction &act_function)
{
    std::stringstream str;
    str << act_function;
    return str.str();
}

/** Formatted output of the PoolingType type. */
inline ::std::ostream &operator<<(::std::ostream &os, const PoolingType &pool_type)
{
    switch(pool_type)
    {
        case PoolingType::AVG:
            os << "AVG";
            break;
        case PoolingType::MAX:
            os << "MAX";
            break;
        case PoolingType::L2:
            os << "L2";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the NormType type. */
inline ::std::ostream &operator<<(::std::ostream &os, const NormType &norm_type)
{
    switch(norm_type)
    {
        case NormType::CROSS_MAP:
            os << "CROSS_MAP";
            break;
        case NormType::IN_MAP_1D:
            os << "IN_MAP_1D";
            break;
        case NormType::IN_MAP_2D:
            os << "IN_MAP_2D";
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
        case EltwiseOperation::ADD:
            os << "ADD";
            break;
        case EltwiseOperation::MUL:
            os << "MUL";
            break;
        case EltwiseOperation::SUB:
            os << "SUB";
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
        case ConvolutionMethod::DEFAULT:
            os << "DEFAULT";
            break;
        case ConvolutionMethod::DIRECT:
            os << "DIRECT";
            break;
        case ConvolutionMethod::GEMM:
            os << "GEMM";
            break;
        case ConvolutionMethod::WINOGRAD:
            os << "WINOGRAD";
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
        case FastMathHint::ENABLED:
            os << "ENABLED";
            break;
        case FastMathHint::DISABLED:
            os << "DISABLED";
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
        case DepthwiseConvolutionMethod::DEFAULT:
            os << "DEFAULT";
            break;
        case DepthwiseConvolutionMethod::GEMV:
            os << "GEMV";
            break;
        case DepthwiseConvolutionMethod::OPTIMIZED_3x3:
            os << "OPTIMIZED_3x3";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the PadStrideInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const PadStrideInfo &pad_stride_info)
{
    os << pad_stride_info.stride().first << "," << pad_stride_info.stride().second;
    os << ";";
    os << pad_stride_info.pad_left() << "," << pad_stride_info.pad_right() << ","
       << pad_stride_info.pad_top() << "," << pad_stride_info.pad_bottom();

    return os;
}

/** Formatted output of the QuantizationInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const QuantizationInfo &quantization_info)
{
    os << "Scale:" << quantization_info.scale << "~"
       << "Offset:" << quantization_info.offset;
    return os;
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_TYPE_PRINTER_H__ */
