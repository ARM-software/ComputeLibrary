/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_TYPE_PRINTER_H__
#define __ARM_COMPUTE_TYPE_PRINTER_H__

#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include <ostream>
#include <sstream>
#include <string>

namespace arm_compute
{
/** Formatted output if arg is not null
 *
 * @param[in] arg Object to print
 *
 * @return String representing arg.
 */
template <typename T>
std::string to_string_if_not_null(T *arg)
{
    if(arg == nullptr)
    {
        return "nullptr";
    }
    else
    {
        return to_string(*arg);
    }
}
/** Formatted output of the Dimensions type.
 *
 * @param[out] os         Output stream.
 * @param[in]  dimensions Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the NonLinearFilterFunction type.
 *
 * @param[out] os       Output stream.
 * @param[in]  function Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the NonLinearFilterFunction type.
 *
 * @param[in] function Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const NonLinearFilterFunction &function)
{
    std::stringstream str;
    str << function;
    return str.str();
}

/** Formatted output of the MatrixPattern type.
 *
 * @param[out] os      Output stream.
 * @param[in]  pattern Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the MatrixPattern type.
 *
 * @param[in] pattern Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const MatrixPattern &pattern)
{
    std::stringstream str;
    str << pattern;
    return str.str();
}

/** Formatted output of the RoundingPolicy type.
 *
 * @param[out] os              Output stream.
 * @param[in]  rounding_policy Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the WeightsInfo type.
 *
 * @param[out] os           Output stream.
 * @param[in]  weights_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const WeightsInfo &weights_info)
{
    os << weights_info.are_reshaped() << ";";
    os << weights_info.num_kernels() << ";" << weights_info.kernel_size().first << "," << weights_info.kernel_size().second;

    return os;
}

/** Formatted output of the ROIPoolingInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  pool_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ROIPoolingLayerInfo &pool_info)
{
    os << pool_info.pooled_width() << "x" << pool_info.pooled_height() << "~" << pool_info.spatial_scale();
    return os;
}

/** Formatted output of the ROIPoolingInfo type.
 *
 * @param[in] pool_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ROIPoolingLayerInfo &pool_info)
{
    std::stringstream str;
    str << pool_info;
    return str.str();
}

/** Formatted output of the BoundingBoxTransformInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  bbox_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const BoundingBoxTransformInfo &bbox_info)
{
    auto weights = bbox_info.weights();
    os << "(" << bbox_info.img_width() << "x" << bbox_info.img_height() << ")~" << bbox_info.scale() << "(weights = {" << weights[0] << ", " << weights[1] << ", " << weights[2] << ", " << weights[3] <<
       "})";
    return os;
}

/** Formatted output of the BoundingBoxTransformInfo type.
 *
 * @param[in] bbox_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const BoundingBoxTransformInfo &bbox_info)
{
    std::stringstream str;
    str << bbox_info;
    return str.str();
}

/** Formatted output of the ComputeAnchorsInfo type.
 *
 * @param[out] os           Output stream.
 * @param[in]  anchors_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ComputeAnchorsInfo &anchors_info)
{
    os << "(" << anchors_info.feat_width() << "x" << anchors_info.feat_height() << ")~" << anchors_info.spatial_scale();
    return os;
}

/** Formatted output of the ComputeAnchorsInfo type.
 *
 * @param[in] anchors_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ComputeAnchorsInfo &anchors_info)
{
    std::stringstream str;
    str << anchors_info;
    return str.str();
}

/** Formatted output of the GenerateProposalsInfo type.
 *
 * @param[out] os             Output stream.
 * @param[in]  proposals_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GenerateProposalsInfo &proposals_info)
{
    os << "(" << proposals_info.im_width() << "x" << proposals_info.im_height() << ")~" << proposals_info.im_scale();
    return os;
}

/** Formatted output of the GenerateProposalsInfo type.
 *
 * @param[in] proposals_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GenerateProposalsInfo &proposals_info)
{
    std::stringstream str;
    str << proposals_info;
    return str.str();
}

/** Formatted output of the QuantizationInfo type.
 *
 * @param[out] os                Output stream.
 * @param[in]  quantization_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const QuantizationInfo &quantization_info)
{
    os << "Scale:" << quantization_info.scale << "~"
       << "Offset:" << quantization_info.offset;
    return os;
}

/** Formatted output of the QuantizationInfo type.
 *
 * @param[in] quantization_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const QuantizationInfo &quantization_info)
{
    std::stringstream str;
    str << quantization_info;
    return str.str();
}

/** Formatted output of the activation function type.
 *
 * @param[out] os           Output stream.
 * @param[in]  act_function Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the activation function info type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::ActivationLayerInfo &info)
{
    std::stringstream str;
    if(info.enabled())
    {
        str << info.activation();
    }
    return str.str();
}

/** Formatted output of the activation function type.
 *
 * @param[in] function Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::ActivationLayerInfo::ActivationFunction &function)
{
    std::stringstream str;
    str << function;
    return str.str();
}

/** Formatted output of the NormType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  norm_type Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of @ref NormalizationLayerInfo.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::NormalizationLayerInfo &info)
{
    std::stringstream str;
    str << info.type() << ":NormSize=" << info.norm_size();
    return str.str();
}

/** Formatted output of @ref NormalizationLayerInfo.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const NormalizationLayerInfo &info)
{
    os << info.type() << ":NormSize=" << info.norm_size();
    return os;
}

/** Formatted output of the PoolingType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  pool_type Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of @ref PoolingLayerInfo.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PoolingLayerInfo &info)
{
    os << info.pool_type();

    return os;
}

/** Formatted output of @ref RoundingPolicy.
 *
 * @param[in] rounding_policy Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const RoundingPolicy &rounding_policy)
{
    std::stringstream str;
    str << rounding_policy;
    return str.str();
}

/** [Print DataLayout type] **/
/** Formatted output of the DataLayout type.
 *
 * @param[out] os          Output stream.
 * @param[in]  data_layout Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DataLayout &data_layout)
{
    switch(data_layout)
    {
        case DataLayout::UNKNOWN:
            os << "UNKNOWN";
            break;
        case DataLayout::NHWC:
            os << "NHWC";
            break;
        case DataLayout::NCHW:
            os << "NCHW";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the DataLayout type.
 *
 * @param[in] data_layout Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::DataLayout &data_layout)
{
    std::stringstream str;
    str << data_layout;
    return str.str();
}
/** [Print DataLayout type] **/

/** Formatted output of the DataLayoutDimension type.
 *
 * @param[out] os              Output stream.
 * @param[in]  data_layout_dim Data layout dimension to print.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DataLayoutDimension &data_layout_dim)
{
    switch(data_layout_dim)
    {
        case DataLayoutDimension::WIDTH:
            os << "WIDTH";
            break;
        case DataLayoutDimension::HEIGHT:
            os << "HEIGHT";
            break;
        case DataLayoutDimension::CHANNEL:
            os << "CHANNEL";
            break;
        case DataLayoutDimension::BATCHES:
            os << "BATCHES";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return os;
}

/** Formatted output of the DataType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  data_type Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the DataType type.
 *
 * @param[in] data_type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::DataType &data_type)
{
    std::stringstream str;
    str << data_type;
    return str.str();
}

/** Formatted output of the Format type.
 *
 * @param[out] os     Output stream.
 * @param[in]  format Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the Format type.
 *
 * @param[in] format Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Format &format)
{
    std::stringstream str;
    str << format;
    return str.str();
}

/** Formatted output of the Channel type.
 *
 * @param[out] os      Output stream.
 * @param[in]  channel Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the Channel type.
 *
 * @param[in] channel Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Channel &channel)
{
    std::stringstream str;
    str << channel;
    return str.str();
}

/** Formatted output of the BorderMode type.
 *
 * @param[out] os   Output stream.
 * @param[in]  mode Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the BorderSize type.
 *
 * @param[out] os     Output stream.
 * @param[in]  border Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const BorderSize &border)
{
    os << border.top << ","
       << border.right << ","
       << border.bottom << ","
       << border.left;

    return os;
}

/** Formatted output of the PaddingList type.
 *
 * @param[out] os      Output stream.
 * @param[in]  padding Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PaddingList &padding)
{
    os << "{";
    for(auto const &p : padding)
    {
        os << "{" << p.first << "," << p.second << "}";
    }
    os << "}";
    return os;
}

/** Formatted output of the Multiples type.
 *
 * @param[out] os        Output stream.
 * @param[in]  multiples Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Multiples &multiples)
{
    os << "(";
    for(size_t i = 0; i < multiples.size() - 1; i++)
    {
        os << multiples[i] << ", ";
    }
    os << multiples.back() << ")";
    return os;
}

/** Formatted output of the InterpolationPolicy type.
 *
 * @param[out] os     Output stream.
 * @param[in]  policy Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the SamplingPolicy type.
 *
 * @param[out] os     Output stream.
 * @param[in]  policy Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the TensorInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const TensorInfo &info)
{
    os << "{Shape=" << info.tensor_shape() << ","
       << "Type=" << info.data_type() << ","
       << "Channels=" << info.num_channels() << "}";
    return os;
}
/** Formatted output of the TensorInfo type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const TensorInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

//FIXME: Check why this doesn't work and the TensorShape and Coordinates overload are needed
/** Formatted output of the Dimensions type.
 *
 * @param[in] dimensions Type to output.
 *
 * @return Formatted string.
 */
template <typename T>
inline std::string to_string(const Dimensions<T> &dimensions)
{
    std::stringstream str;
    str << dimensions;
    return str.str();
}

/** Formatted output of the Strides type.
 *
 * @param[in] stride Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Strides &stride)
{
    std::stringstream str;
    str << stride;
    return str.str();
}

/** Formatted output of the TensorShape type.
 *
 * @param[in] shape Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const TensorShape &shape)
{
    std::stringstream str;
    str << shape;
    return str.str();
}

/** Formatted output of the Coordinates type.
 *
 * @param[in] coord Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Coordinates &coord)
{
    std::stringstream str;
    str << coord;
    return str.str();
}

/** Formatted output of the GEMMReshapeInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMReshapeInfo &info)
{
    os << "{m=" << info.m() << ",";
    os << "n=" << info.n() << ",";
    os << "k=" << info.k() << ",";
    os << "mult_transpose1xW_width=" << info.mult_transpose1xW_width() << ",";
    os << "mult_interleave4x4_height=" << info.mult_interleave4x4_height();
    os << "}";

    return os;
}

/** Formatted output of the GEMMInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMInfo &info)
{
    os << "{is_a_reshaped=" << info.is_a_reshaped() << ",";
    os << "is_b_reshaped=" << info.is_b_reshaped() << ",";
    os << "reshape_b_only_on_first_run=" << info.reshape_b_only_on_first_run() << ",";
    os << "}";

    return os;
}

/** Formatted output of the Window::Dimension type.
 *
 * @param[out] os  Output stream.
 * @param[in]  dim Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Window::Dimension &dim)
{
    os << "{start=" << dim.start() << ", end=" << dim.end() << ", step=" << dim.step() << "}";

    return os;
}
/** Formatted output of the Window type.
 *
 * @param[out] os  Output stream.
 * @param[in]  win Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Window &win)
{
    os << "{";
    for(unsigned int i = 0; i < Coordinates::num_max_dimensions; i++)
    {
        if(i > 0)
        {
            os << ", ";
        }
        os << win[i];
    }
    os << "}";

    return os;
}

/** Formatted output of the WeightsInfo type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const WeightsInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the GEMMReshapeInfo type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GEMMReshapeInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the GEMMInfo type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GEMMInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the Window::Dimension type.
 *
 * @param[in] dim Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Window::Dimension &dim)
{
    std::stringstream str;
    str << dim;
    return str.str();
}
/** Formatted output of the Window type.
 *
 * @param[in] win Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Window &win)
{
    std::stringstream str;
    str << win;
    return str.str();
}

/** Formatted output of the Rectangle type.
 *
 * @param[out] os   Output stream.
 * @param[in]  rect Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Rectangle &rect)
{
    os << rect.width << "x" << rect.height;
    os << "+" << rect.x << "+" << rect.y;

    return os;
}

/** Formatted output of the PadStrideInfo type.
 *
 * @param[out] os              Output stream.
 * @param[in]  pad_stride_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PadStrideInfo &pad_stride_info)
{
    os << pad_stride_info.stride().first << "," << pad_stride_info.stride().second;
    os << ";";
    os << pad_stride_info.pad_left() << "," << pad_stride_info.pad_right() << ","
       << pad_stride_info.pad_top() << "," << pad_stride_info.pad_bottom();

    return os;
}

/** Formatted output of the PadStrideInfo type.
 *
 * @param[in] pad_stride_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PadStrideInfo &pad_stride_info)
{
    std::stringstream str;
    str << pad_stride_info;
    return str.str();
}

/** Formatted output of the BorderMode type.
 *
 * @param[in] mode Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const BorderMode &mode)
{
    std::stringstream str;
    str << mode;
    return str.str();
}

/** Formatted output of the BorderSize type.
 *
 * @param[in] border Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const BorderSize &border)
{
    std::stringstream str;
    str << border;
    return str.str();
}

/** Formatted output of the PaddingList type.
 *
 * @param[in] padding Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PaddingList &padding)
{
    std::stringstream str;
    str << padding;
    return str.str();
}

/** Formatted output of the Multiples type.
 *
 * @param[in] multiples Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Multiples &multiples)
{
    std::stringstream str;
    str << multiples;
    return str.str();
}

/** Formatted output of the InterpolationPolicy type.
 *
 * @param[in] policy Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const InterpolationPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

/** Formatted output of the SamplingPolicy type.
 *
 * @param[in] policy Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const SamplingPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

/** Formatted output of the ConvertPolicy type.
 *
 * @param[out] os     Output stream.
 * @param[in]  policy Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the ArithmeticOperation type.
 *
 * @param[out] os Output stream.
 * @param[in]  op Operation to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ArithmeticOperation &op)
{
    switch(op)
    {
        case ArithmeticOperation::ADD:
            os << "ADD";
            break;
        case ArithmeticOperation::SUB:
            os << "SUB";
            break;
        case ArithmeticOperation::DIV:
            os << "DIV";
            break;
        case ArithmeticOperation::MAX:
            os << "MAX";
            break;
        case ArithmeticOperation::MIN:
            os << "MIN";
            break;
        case ArithmeticOperation::SQUARED_DIFF:
            os << "SQUARED_DIFF";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Arithmetic Operation
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ArithmeticOperation &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Reduction Operations.
 *
 * @param[out] os Output stream.
 * @param[in]  op Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ReductionOperation &op)
{
    switch(op)
    {
        case ReductionOperation::SUM:
            os << "SUM";
            break;
        case ReductionOperation::SUM_SQUARE:
            os << "SUM_SQUARE";
            break;
        case ReductionOperation::MEAN_SUM:
            os << "MEAN_SUM";
            break;
        case ReductionOperation::ARG_IDX_MAX:
            os << "ARG_IDX_MAX";
            break;
        case ReductionOperation::ARG_IDX_MIN:
            os << "ARG_IDX_MIN";
            break;
        case ReductionOperation::PROD:
            os << "PROD";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Reduction Operations.
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ReductionOperation &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Comparison Operations.
 *
 * @param[out] os Output stream.
 * @param[in]  op Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ComparisonOperation &op)
{
    switch(op)
    {
        case ComparisonOperation::Equal:
            os << "Equal";
            break;
        case ComparisonOperation::NotEqual:
            os << "NotEqual";
            break;
        case ComparisonOperation::Greater:
            os << "Greater";
            break;
        case ComparisonOperation::GreaterEqual:
            os << "GreaterEqual";
            break;
        case ComparisonOperation::Less:
            os << "Less";
            break;
        case ComparisonOperation::LessEqual:
            os << "LessEqual";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Elementwise unary Operations.
 *
 * @param[out] os Output stream.
 * @param[in]  op Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ElementWiseUnary &op)
{
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            os << "RSQRT";
            break;
        case ElementWiseUnary::EXP:
            os << "EXP";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Comparison Operations.
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ComparisonOperation &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Elementwise unary Operations.
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ElementWiseUnary &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Norm Type.
 *
 * @param[in] type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const NormType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Pooling Type.
 *
 * @param[in] type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PoolingType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Pooling Layer Info.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PoolingLayerInfo &info)
{
    std::stringstream str;
    str << "{Type=" << info.pool_type() << ","
        << "IsGlobalPooling=" << info.is_global_pooling();
    if(!info.is_global_pooling())
    {
        str << ","
            << "PoolSize=" << info.pool_size().width << "," << info.pool_size().height << ","
            << "PadStride=" << info.pad_stride_info();
    }
    str << "}";
    return str.str();
}

/** Formatted output of the PriorBoxLayerInfo.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PriorBoxLayerInfo &info)
{
    std::stringstream str;
    str << "{";
    str << "Clip:" << info.clip()
        << "Flip:" << info.flip()
        << "StepX:" << info.steps()[0]
        << "StepY:" << info.steps()[1]
        << "MinSizes:" << info.min_sizes().size()
        << "MaxSizes:" << info.max_sizes().size()
        << "ImgSizeX:" << info.img_size().x
        << "ImgSizeY:" << info.img_size().y
        << "Offset:" << info.offset()
        << "Variances:" << info.variances().size();
    str << "}";
    return str.str();
}

/** Formatted output of the KeyPoint type.
 *
 * @param[out] os    Output stream
 * @param[in]  point Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the PhaseType type.
 *
 * @param[out] os         Output stream
 * @param[in]  phase_type Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the PhaseType type.
 *
 * @param[in] type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::PhaseType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the MagnitudeType type.
 *
 * @param[out] os             Output stream
 * @param[in]  magnitude_type Type to output.
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the MagnitudeType type.
 *
 * @param[in] type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::MagnitudeType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the HOGNormType type.
 *
 * @param[out] os        Output stream
 * @param[in]  norm_type Type to output
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the HOGNormType type.
 *
 * @param[in] type Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const HOGNormType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Size2D type.
 *
 * @param[out] os   Output stream
 * @param[in]  size Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Size2D &size)
{
    os << size.width << "x" << size.height;

    return os;
}

/** Formatted output of the Size2D type.
 *
 * @param[in] type Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const Size2D &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the HOGInfo type.
 *
 * @param[out] os       Output stream
 * @param[in]  hog_info Type to output
 *
 * @return Modified output stream.
 */
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

/** Formatted output of the HOGInfo type.
 *
 * @param[in] type Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const HOGInfo &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the ConvolutionMethod type.
 *
 * @param[out] os          Output stream
 * @param[in]  conv_method Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ConvolutionMethod &conv_method)
{
    switch(conv_method)
    {
        case ConvolutionMethod::GEMM:
            os << "GEMM";
            break;
        case ConvolutionMethod::DIRECT:
            os << "DIRECT";
            break;
        case ConvolutionMethod::WINOGRAD:
            os << "WINOGRAD";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the ConvolutionMethod type.
 *
 * @param[in] conv_method Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const ConvolutionMethod &conv_method)
{
    std::stringstream str;
    str << conv_method;
    return str.str();
}

/** Formatted output of the GPUTarget type.
 *
 * @param[out] os         Output stream
 * @param[in]  gpu_target Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GPUTarget &gpu_target)
{
    switch(gpu_target)
    {
        case GPUTarget::GPU_ARCH_MASK:
            os << "GPU_ARCH_MASK";
            break;
        case GPUTarget::MIDGARD:
            os << "MIDGARD";
            break;
        case GPUTarget::BIFROST:
            os << "BIFROST";
            break;
        case GPUTarget::T600:
            os << "T600";
            break;
        case GPUTarget::T700:
            os << "T700";
            break;
        case GPUTarget::T800:
            os << "T800";
            break;
        case GPUTarget::G71:
            os << "G71";
            break;
        case GPUTarget::G72:
            os << "G72";
            break;
        case GPUTarget::G51:
            os << "G51";
            break;
        case GPUTarget::G51BIG:
            os << "G51BIG";
            break;
        case GPUTarget::G51LIT:
            os << "G51LIT";
            break;
        case GPUTarget::G76:
            os << "G76";
            break;
        case GPUTarget::TTRX:
            os << "TTRX";
            break;
        case GPUTarget::TBOX:
            os << "TBOX";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the GPUTarget type.
 *
 * @param[in] gpu_target Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const GPUTarget &gpu_target)
{
    std::stringstream str;
    str << gpu_target;
    return str.str();
}

/** Formatted output of the DetectionWindow type.
 *
 * @param[out] os               Output stream
 * @param[in]  detection_window Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DetectionWindow &detection_window)
{
    os << "{x=" << detection_window.x << ","
       << "y=" << detection_window.y << ","
       << "width=" << detection_window.width << ","
       << "height=" << detection_window.height << ","
       << "idx_class=" << detection_window.idx_class << ","
       << "score=" << detection_window.score << "}";

    return os;
}

/** Formatted output of the DetectionOutputLayerCodeType type.
 *
 * @param[out] os             Output stream
 * @param[in]  detection_code Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DetectionOutputLayerCodeType &detection_code)
{
    switch(detection_code)
    {
        case DetectionOutputLayerCodeType::CENTER_SIZE:
            os << "CENTER_SIZE";
            break;
        case DetectionOutputLayerCodeType::CORNER:
            os << "CORNER";
            break;
        case DetectionOutputLayerCodeType::CORNER_SIZE:
            os << "CORNER_SIZE";
            break;
        case DetectionOutputLayerCodeType::TF_CENTER:
            os << "TF_CENTER";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}
/** Formatted output of the DetectionOutputLayerCodeType type.
 *
 * @param[in] detection_code Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const DetectionOutputLayerCodeType &detection_code)
{
    std::stringstream str;
    str << detection_code;
    return str.str();
}

/** Formatted output of the DetectionOutputLayerInfo type.
 *
 * @param[out] os             Output stream
 * @param[in]  detection_info Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DetectionOutputLayerInfo &detection_info)
{
    os << "{Classes=" << detection_info.num_classes() << ","
       << "ShareLocation=" << detection_info.share_location() << ","
       << "CodeType=" << detection_info.code_type() << ","
       << "VarianceEncodedInTarget=" << detection_info.variance_encoded_in_target() << ","
       << "KeepTopK=" << detection_info.keep_top_k() << ","
       << "NMSThreshold=" << detection_info.nms_threshold() << ","
       << "Eta=" << detection_info.eta() << ","
       << "BackgroundLabelId=" << detection_info.background_label_id() << ","
       << "ConfidenceThreshold=" << detection_info.confidence_threshold() << ","
       << "TopK=" << detection_info.top_k() << ","
       << "NumLocClasses=" << detection_info.num_loc_classes()
       << "}";

    return os;
}

/** Formatted output of the DetectionOutputLayerInfo type.
 *
 * @param[in] detection_info Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const DetectionOutputLayerInfo &detection_info)
{
    std::stringstream str;
    str << detection_info;
    return str.str();
}
/** Formatted output of the DetectionWindow type.
 *
 * @param[in] detection_window Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const DetectionWindow &detection_window)
{
    std::stringstream str;
    str << detection_window;
    return str.str();
}

/** Formatted output of the Termination type.
 *
 * @param[out] os          Output stream
 * @param[in]  termination Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Termination &termination)
{
    switch(termination)
    {
        case Termination::TERM_CRITERIA_EPSILON:
            os << "TERM_CRITERIA_EPSILON";
            break;
        case Termination::TERM_CRITERIA_ITERATIONS:
            os << "TERM_CRITERIA_ITERATIONS";
            break;
        case Termination::TERM_CRITERIA_BOTH:
            os << "TERM_CRITERIA_BOTH";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Termination type.
 *
 * @param[in] termination Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const Termination &termination)
{
    std::stringstream str;
    str << termination;
    return str.str();
}

/** Formatted output of the CPUModel type.
 *
 * @param[out] os        Output stream
 * @param[in]  cpu_model Model to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const CPUModel &cpu_model)
{
    switch(cpu_model)
    {
        case CPUModel::GENERIC:
            os << "GENERIC";
            break;
        case CPUModel::GENERIC_FP16:
            os << "GENERIC_FP16";
            break;
        case CPUModel::GENERIC_FP16_DOT:
            os << "GENERIC_FP16_DOT";
            break;
        case CPUModel::A53:
            os << "A53";
            break;
        case CPUModel::A55r0:
            os << "A55r0";
            break;
        case CPUModel::A55r1:
            os << "A55r1";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the CPUModel type.
 *
 * @param[in] cpu_model Model to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const CPUModel &cpu_model)
{
    std::stringstream str;
    str << cpu_model;
    return str.str();
}
/** Formatted output of a vector of objects.
 *
 * @param[out] os   Output stream
 * @param[in]  args Vector of objects to print
 *
 * @return Modified output stream.
 */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const std::vector<T> &args)
{
    os << "[";
    bool first = true;
    for(auto &arg : args)
    {
        if(first)
        {
            first = false;
        }
        else
        {
            os << ", ";
        }
        os << arg;
    }
    os << "]";
    return os;
}

/** Formatted output of @ref PriorBoxLayerInfo.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PriorBoxLayerInfo &info)
{
    os << "Clip:" << info.clip()
       << "Flip:" << info.flip()
       << "StepX:" << info.steps()[0]
       << "StepY:" << info.steps()[1]
       << "MinSizes:" << info.min_sizes()
       << "MaxSizes:" << info.max_sizes()
       << "ImgSizeX:" << info.img_size().x
       << "ImgSizeY:" << info.img_size().y
       << "Offset:" << info.offset()
       << "Variances:" << info.variances();

    return os;
}

/** Formatted output of a vector of objects.
 *
 * @param[in] args Vector of objects to print
 *
 * @return String representing args.
 */
template <typename T>
std::string to_string(const std::vector<T> &args)
{
    std::stringstream str;
    str << args;
    return str.str();
}

/** Formatted output of the WinogradInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const WinogradInfo &info)
{
    os << "{OutputTileSize=" << info.output_tile_size << ","
       << "KernelSize=" << info.kernel_size << ","
       << "PadStride=" << info.convolution_info << ","
       << "OutputDataLayout=" << info.output_data_layout << "}";

    return os;
}

inline std::string to_string(const WinogradInfo &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Fallback method: try to use std::to_string:
 *
 * @param[in] val Value to convert to string
 *
 * @return String representing val.
 */
template <typename T>
inline std::string to_string(const T &val)
{
    return support::cpp11::to_string(val);
}

} // namespace arm_compute

#endif /* __ARM_COMPUTE_TYPE_PRINTER_H__ */
