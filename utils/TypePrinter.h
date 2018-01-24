/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "tests/Types.h"

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

/** Formatted output of the WeightsInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const WeightsInfo &weights_info)
{
    os << weights_info.are_reshaped() << ";";
    os << weights_info.num_kernels() << ";" << weights_info.kernel_size().first << "," << weights_info.kernel_size().second;

    return os;
}

/** Formatted output of the ROIPoolingInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const ROIPoolingLayerInfo &pool_info)
{
    os << pool_info.pooled_width() << "x" << pool_info.pooled_height() << "~" << pool_info.spatial_scale();
    return os;
}

/** Formatted output of the QuantizationInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const QuantizationInfo &quantization_info)
{
    os << "Scale:" << quantization_info.scale << "~"
       << "Offset:" << quantization_info.offset;
    return os;
}

inline std::string to_string(const QuantizationInfo &quantization_info)
{
    std::stringstream str;
    str << quantization_info;
    return str.str();
}

inline ::std::ostream &operator<<(::std::ostream &os, const FixedPointOp &op)
{
    switch(op)
    {
        case FixedPointOp::ADD:
            os << "ADD";
            break;
        case FixedPointOp::SUB:
            os << "SUB";
            break;
        case FixedPointOp::MUL:
            os << "MUL";
            break;
        case FixedPointOp::EXP:
            os << "EXP";
            break;
        case FixedPointOp::LOG:
            os << "LOG";
            break;
        case FixedPointOp::INV_SQRT:
            os << "INV_SQRT";
            break;
        case FixedPointOp::RECIPROCAL:
            os << "RECIPROCAL";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const FixedPointOp &op)
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

inline std::string to_string(const arm_compute::ActivationLayerInfo &info)
{
    std::stringstream str;
    str << info.activation();
    return str.str();
}

inline std::string to_string(const arm_compute::ActivationLayerInfo::ActivationFunction &function)
{
    std::stringstream str;
    str << function;
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

inline std::string to_string(const arm_compute::NormalizationLayerInfo &info)
{
    std::stringstream str;
    str << info.type();
    return str.str();
}

/** Formatted output of @ref NormalizationLayerInfo. */
inline ::std::ostream &operator<<(::std::ostream &os, const NormalizationLayerInfo &info)
{
    os << info.type();
    return os;
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

/** Formatted output of @ref PoolingLayerInfo. */
inline ::std::ostream &operator<<(::std::ostream &os, const PoolingLayerInfo &info)
{
    os << info.pool_type();

    return os;
}

inline std::string to_string(const RoundingPolicy &rounding_policy)
{
    std::stringstream str;
    str << rounding_policy;
    return str.str();
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

inline std::string to_string(const arm_compute::DataType &data_type)
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

inline std::string to_string(const Format &format)
{
    std::stringstream str;
    str << format;
    return str.str();
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

/** Formatted output of the BorderSize type. */
inline ::std::ostream &operator<<(::std::ostream &os, const BorderSize &border)
{
    os << border.top << ","
       << border.right << ","
       << border.bottom << ","
       << border.left;

    return os;
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

/** Formatted output of the SamplingPolicy type. */
inline ::std::ostream &operator<<(::std::ostream &os, const SamplingPolicy &policy)
{
    switch(policy)
    {
        case SamplingPolicy::CENTER:
            os << "CENTER";
            break;
        case SamplingPolicy::TOP_LEFT:
            os << "TOP_LEFT";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the TensorInfo type. */
inline std::string to_string(const TensorInfo &info)
{
    std::stringstream str;
    str << "{Shape=" << info.tensor_shape() << ","
        << "Type=" << info.data_type() << ","
        << "Channels=" << info.num_channels() << ","
        << "FixedPointPos=" << info.fixed_point_position() << "}";
    return str.str();
}

template <typename T>
inline std::string to_string(const Dimensions<T> &dimensions)
{
    std::stringstream str;
    str << dimensions;
    return str.str();
}

inline std::string to_string(const Strides &stride)
{
    std::stringstream str;
    str << stride;
    return str.str();
}

/** Formatted output of the TensorShape type. */
inline std::string to_string(const TensorShape &shape)
{
    std::stringstream str;
    str << shape;
    return str.str();
}

/** Formatted output of the Coordinates type. */
inline std::string to_string(const Coordinates &coord)
{
    std::stringstream str;
    str << coord;
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
    os << pad_stride_info.pad_left() << "," << pad_stride_info.pad_right() << ","
       << pad_stride_info.pad_top() << "," << pad_stride_info.pad_bottom();

    return os;
}

inline std::string to_string(const PadStrideInfo &pad_stride_info)
{
    std::stringstream str;
    str << pad_stride_info;
    return str.str();
}

inline std::string to_string(const BorderMode &mode)
{
    std::stringstream str;
    str << mode;
    return str.str();
}

inline std::string to_string(const BorderSize &border)
{
    std::stringstream str;
    str << border;
    return str.str();
}

inline std::string to_string(const InterpolationPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

inline std::string to_string(const SamplingPolicy &policy)
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

inline std::string to_string(const NormType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

inline std::string to_string(const PoolingType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

inline std::string to_string(const PoolingLayerInfo &info)
{
    std::stringstream str;
    str << "{Type=" << info.pool_type() << ","
        << "IsGlobalPooling=" << info.is_global_pooling();
    if(!info.is_global_pooling())
    {
        str << ","
            << "PoolSize=" << info.pool_size() << ","
            << "PadStride=" << info.pad_stride_info();
    }
    str << "}";
    return str.str();
}

/** Formatted output of the KeyPoint type. */
inline ::std::ostream &operator<<(::std::ostream &os, const KeyPoint &point)
{
    os << "{x=" << point.x << ","
       << "y=" << point.y << ","
       << "strength=" << point.strength << ","
       << "scale=" << point.scale << ","
       << "orientation=" << point.orientation << ","
       << "tracking_status=" << point.tracking_status << ","
       << "error=" << point.error << "}";

    return os;
}

/** Formatted output of the PhaseType type. */
inline ::std::ostream &operator<<(::std::ostream &os, const PhaseType &phase_type)
{
    switch(phase_type)
    {
        case PhaseType::SIGNED:
            os << "SIGNED";
            break;
        case PhaseType::UNSIGNED:
            os << "UNSIGNED";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const arm_compute::PhaseType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the MagnitudeType type. */
inline ::std::ostream &operator<<(::std::ostream &os, const MagnitudeType &magnitude_type)
{
    switch(magnitude_type)
    {
        case MagnitudeType::L1NORM:
            os << "L1NORM";
            break;
        case MagnitudeType::L2NORM:
            os << "L2NORM";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const arm_compute::MagnitudeType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the GradientDimension type. */
inline ::std::ostream &operator<<(::std::ostream &os, const GradientDimension &dim)
{
    switch(dim)
    {
        case GradientDimension::GRAD_X:
            os << "GRAD_X";
            break;
        case GradientDimension::GRAD_Y:
            os << "GRAD_Y";
            break;
        case GradientDimension::GRAD_XY:
            os << "GRAD_XY";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const arm_compute::GradientDimension &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the HOGNormType type. */
inline ::std::ostream &operator<<(::std::ostream &os, const HOGNormType &norm_type)
{
    switch(norm_type)
    {
        case HOGNormType::L1_NORM:
            os << "L1_NORM";
            break;
        case HOGNormType::L2_NORM:
            os << "L2_NORM";
            break;
        case HOGNormType::L2HYS_NORM:
            os << "L2HYS_NORM";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const HOGNormType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Size2D type. */
inline ::std::ostream &operator<<(::std::ostream &os, const Size2D &size)
{
    os << size.width << "x" << size.height;

    return os;
}

inline std::string to_string(const Size2D &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Size2D type. */
inline ::std::ostream &operator<<(::std::ostream &os, const HOGInfo &hog_info)
{
    os << "{CellSize=" << hog_info.cell_size() << ","
       << "BlockSize=" << hog_info.block_size() << ","
       << "DetectionWindowSize=" << hog_info.detection_window_size() << ","
       << "BlockStride=" << hog_info.block_stride() << ","
       << "NumBins=" << hog_info.num_bins() << ","
       << "NormType=" << hog_info.normalization_type() << ","
       << "L2HystThreshold=" << hog_info.l2_hyst_threshold() << ","
       << "PhaseType=" << hog_info.phase_type() << "}";

    return os;
}

/** Formatted output of the HOGInfo type. */
inline std::string to_string(const HOGInfo &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_TYPE_PRINTER_H__ */
