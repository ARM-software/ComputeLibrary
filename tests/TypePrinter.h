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
#ifndef __ARM_COMPUTE_TEST_TYPE_PRINTER_H__
#define __ARM_COMPUTE_TEST_TYPE_PRINTER_H__

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

#include <ostream>
#include <sstream>
#include <string>

namespace arm_compute
{
/** Formatted output of the Dimensions type. */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const Dimensions<T> &dimensions)
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

//FIXME: Check why this doesn't work and the TensorShape overload is needed
template <typename T>
inline std::string to_string(const Dimensions<T> &dimensions)
{
    std::stringstream str;
    str << dimensions;
    return str.str();
}

inline std::string to_string(const TensorShape &shape)
{
    std::stringstream str;
    str << shape;
    return str.str();
}

/** Formatted output of the Rectangle type. */
inline ::std::ostream &operator<<(::std::ostream &os, const Rectangle &rect)
{
    os << rect.width << "x" << rect.height;
    os << "+" << rect.x << "+" << rect.y;

    return os;
}

/** Formatted output of the PadStridInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const PadStrideInfo &pad_stride_info)
{
    os << pad_stride_info.stride().first << "," << pad_stride_info.stride().second;
    os << ";";
    os << pad_stride_info.pad().first << "," << pad_stride_info.pad().second;

    return os;
}

inline std::string to_string(const PadStrideInfo &pad_stride_info)
{
    std::stringstream str;
    str << pad_stride_info;
    return str.str();
}

/** Formatted output of the ROIPoolingInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const ROIPoolingLayerInfo &pool_info)
{
    os << pool_info.pooled_width() << "x" << pool_info.pooled_height() << "~" << pool_info.spatial_scale();
    return os;
}

/** Formatted output of the BorderMode type. */
inline ::std::ostream &operator<<(::std::ostream &os, const BorderMode &mode)
{
    switch(mode)
    {
        case BorderMode::UNDEFINED:
            os << "UNDEFINED";
            break;
        case BorderMode::CONSTANT:
            os << "CONSTANT";
            break;
        case BorderMode::REPLICATE:
            os << "REPLICATE";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const BorderMode &mode)
{
    std::stringstream str;
    str << mode;
    return str.str();
}

/** Formatted output of the NonLinearFilterFunction type. */
inline ::std::ostream &operator<<(::std::ostream &os, const NonLinearFilterFunction &function)
{
    switch(function)
    {
        case NonLinearFilterFunction::MAX:
            os << "MAX";
            break;
        case NonLinearFilterFunction::MEDIAN:
            os << "MEDIAN";
            break;
        case NonLinearFilterFunction::MIN:
            os << "MIN";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const NonLinearFilterFunction &function)
{
    std::stringstream str;
    str << function;
    return str.str();
}

/** Formatted output of the MatrixPattern type. */
inline ::std::ostream &operator<<(::std::ostream &os, const MatrixPattern &pattern)
{
    switch(pattern)
    {
        case MatrixPattern::BOX:
            os << "BOX";
            break;
        case MatrixPattern::CROSS:
            os << "CROSS";
            break;
        case MatrixPattern::DISK:
            os << "DISK";
            break;
        case MatrixPattern::OTHER:
            os << "OTHER";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const MatrixPattern &pattern)
{
    std::stringstream str;
    str << pattern;
    return str.str();
}

/** Formatted output of the InterpolationPolicy type. */
inline ::std::ostream &operator<<(::std::ostream &os, const InterpolationPolicy &policy)
{
    switch(policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
            os << "NEAREST_NEIGHBOR";
            break;
        case InterpolationPolicy::BILINEAR:
            os << "BILINEAR";
            break;
        case InterpolationPolicy::AREA:
            os << "AREA";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const InterpolationPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

/** Formatted output of the ConversionPolicy type. */
inline ::std::ostream &operator<<(::std::ostream &os, const ConvertPolicy &policy)
{
    switch(policy)
    {
        case ConvertPolicy::WRAP:
            os << "WRAP";
            break;
        case ConvertPolicy::SATURATE:
            os << "SATURATE";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const ConvertPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

/** Formatted output of the Reduction Operations. */
inline ::std::ostream &operator<<(::std::ostream &os, const ReductionOperation &op)
{
    switch(op)
    {
        case ReductionOperation::SUM_SQUARE:
            os << "SUM_SQUARE";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const ReductionOperation &op)
{
    std::stringstream str;
    str << op;
    return str.str();
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
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            os << "LU_BOUNDED_RELU";
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

inline std::string to_string(const ActivationLayerInfo::ActivationFunction &function)
{
    std::stringstream str;
    str << function;
    return str.str();
}

inline std::string to_string(const ActivationLayerInfo &info)
{
    std::stringstream str;
    str << info.activation();
    return str.str();
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

inline std::string to_string(const NormType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

inline std::string to_string(const NormalizationLayerInfo &info)
{
    std::stringstream str;
    str << info.type();
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
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const PoolingType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of @ref PoolingLayerInfo. */
inline ::std::ostream &operator<<(::std::ostream &os, const PoolingLayerInfo &info)
{
    os << info.pool_type() << ";" << info.pool_size() << ";" << info.pad_stride_info();

    return os;
}

inline std::string to_string(const PoolingLayerInfo &info)
{
    std::stringstream str;
    str << info.pool_type();
    return str.str();
}

/** Formatted output of the RoundingPolicy type. */
inline ::std::ostream &operator<<(::std::ostream &os, const RoundingPolicy &rounding_policy)
{
    switch(rounding_policy)
    {
        case RoundingPolicy::TO_ZERO:
            os << "TO_ZERO";
            break;
        case RoundingPolicy::TO_NEAREST_UP:
            os << "TO_NEAREST_UP";
            break;
        case RoundingPolicy::TO_NEAREST_EVEN:
            os << "TO_NEAREST_EVEN";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

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

inline std::string to_string(const DataType &data_type)
{
    std::stringstream str;
    str << data_type;
    return str.str();
}

/** Formatted output of the Format type. */
inline ::std::ostream &operator<<(::std::ostream &os, const Format &format)
{
    switch(format)
    {
        case Format::UNKNOWN:
            os << "UNKNOWN";
            break;
        case Format::U8:
            os << "U8";
            break;
        case Format::S16:
            os << "S16";
            break;
        case Format::U16:
            os << "U16";
            break;
        case Format::S32:
            os << "S32";
            break;
        case Format::U32:
            os << "U32";
            break;
        case Format::F16:
            os << "F16";
            break;
        case Format::F32:
            os << "F32";
            break;
        case Format::UV88:
            os << "UV88";
            break;
        case Format::RGB888:
            os << "RGB888";
            break;
        case Format::RGBA8888:
            os << "RGBA8888";
            break;
        case Format::YUV444:
            os << "YUV444";
            break;
        case Format::YUYV422:
            os << "YUYV422";
            break;
        case Format::NV12:
            os << "NV12";
            break;
        case Format::NV21:
            os << "NV21";
            break;
        case Format::IYUV:
            os << "IYUV";
            break;
        case Format::UYVY422:
            os << "UYVY422";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Channel type. */
inline ::std::ostream &operator<<(::std::ostream &os, const Channel &channel)
{
    switch(channel)
    {
        case Channel::UNKNOWN:
            os << "UNKNOWN";
            break;
        case Channel::C0:
            os << "C0";
            break;
        case Channel::C1:
            os << "C1";
            break;
        case Channel::C2:
            os << "C2";
            break;
        case Channel::C3:
            os << "C3";
            break;
        case Channel::R:
            os << "R";
            break;
        case Channel::G:
            os << "G";
            break;
        case Channel::B:
            os << "B";
            break;
        case Channel::A:
            os << "A";
            break;
        case Channel::Y:
            os << "Y";
            break;
        case Channel::U:
            os << "U";
            break;
        case Channel::V:
            os << "V";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the BorderSize type. */
inline ::std::ostream &operator<<(::std::ostream &os, const BorderSize &border)
{
    os << border.top << ","
       << border.right << ","
       << border.bottom << ","
       << border.left;

    return os;
}
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_TYPE_PRINTER_H__ */
