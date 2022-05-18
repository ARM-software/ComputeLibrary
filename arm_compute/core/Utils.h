/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_H
#define ARM_COMPUTE_UTILS_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Rounding.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Version.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Calculate the rounded up quotient of val / m.
 *
 * @param[in] val Value to divide and round up.
 * @param[in] m   Value to divide by.
 *
 * @return the result.
 */
template <typename S, typename T>
constexpr auto DIV_CEIL(S val, T m) -> decltype((val + m - 1) / m)
{
    return (val + m - 1) / m;
}

/** Computes the smallest number larger or equal to value that is a multiple of divisor.
 *
 * @param[in] value   Lower bound value
 * @param[in] divisor Value to compute multiple of.
 *
 * @return the result.
 */
template <typename S, typename T>
inline auto ceil_to_multiple(S value, T divisor) -> decltype(((value + divisor - 1) / divisor) * divisor)
{
    ARM_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
    return DIV_CEIL(value, divisor) * divisor;
}

/** Computes the largest number smaller or equal to value that is a multiple of divisor.
 *
 * @param[in] value   Upper bound value
 * @param[in] divisor Value to compute multiple of.
 *
 * @return the result.
 */
template <typename S, typename T>
inline auto floor_to_multiple(S value, T divisor) -> decltype((value / divisor) * divisor)
{
    ARM_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
    return (value / divisor) * divisor;
}

/** Load an entire file in memory
 *
 * @param[in] filename Name of the file to read.
 * @param[in] binary   Is it a binary file ?
 *
 * @return The content of the file.
 */
std::string read_file(const std::string &filename, bool binary);

/** The size in bytes of the data type
 *
 * @param[in] data_type Input data type
 *
 * @return The size in bytes of the data type
 */
inline size_t data_size_from_type(DataType data_type)
{
    switch(data_type)
    {
        case DataType::U8:
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            return 1;
        case DataType::U16:
        case DataType::S16:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
        case DataType::BFLOAT16:
        case DataType::F16:
            return 2;
        case DataType::F32:
        case DataType::U32:
        case DataType::S32:
            return 4;
        case DataType::F64:
        case DataType::U64:
        case DataType::S64:
            return 8;
        case DataType::SIZET:
            return sizeof(size_t);
        default:
            ARM_COMPUTE_ERROR("Invalid data type");
            return 0;
    }
}

/** The size in bytes of the pixel format
 *
 * @param[in] format Input format
 *
 * @return The size in bytes of the pixel format
 */
inline size_t pixel_size_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
            return 1;
        case Format::U16:
        case Format::S16:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::UV88:
        case Format::YUYV422:
        case Format::UYVY422:
            return 2;
        case Format::RGB888:
            return 3;
        case Format::RGBA8888:
            return 4;
        case Format::U32:
        case Format::S32:
        case Format::F32:
            return 4;
        //Doesn't make sense for planar formats:
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        case Format::YUV444:
        default:
            ARM_COMPUTE_ERROR("Undefined pixel size for given format");
            return 0;
    }
}

/** The size in bytes of the data type
 *
 * @param[in] dt Input data type
 *
 * @return The size in bytes of the data type
 */
inline size_t element_size_from_data_type(DataType dt)
{
    switch(dt)
    {
        case DataType::S8:
        case DataType::U8:
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            return 1;
        case DataType::U16:
        case DataType::S16:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
        case DataType::BFLOAT16:
        case DataType::F16:
            return 2;
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            return 4;
        default:
            ARM_COMPUTE_ERROR("Undefined element size for given data type");
            return 0;
    }
}

/** Return the data type used by a given single-planar pixel format
 *
 * @param[in] format Input format
 *
 * @return The size in bytes of the pixel format
 */
inline DataType data_type_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
        case Format::UV88:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
            return DataType::U8;
        case Format::U16:
            return DataType::U16;
        case Format::S16:
            return DataType::S16;
        case Format::U32:
            return DataType::U32;
        case Format::S32:
            return DataType::S32;
        case Format::BFLOAT16:
            return DataType::BFLOAT16;
        case Format::F16:
            return DataType::F16;
        case Format::F32:
            return DataType::F32;
        //Doesn't make sense for planar formats:
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        case Format::YUV444:
        default:
            ARM_COMPUTE_ERROR("Not supported data_type for given format");
            return DataType::UNKNOWN;
    }
}

/** Return the plane index of a given channel given an input format.
 *
 * @param[in] format  Input format
 * @param[in] channel Input channel
 *
 * @return The plane index of the specific channel of the specific format
 */
inline int plane_idx_from_channel(Format format, Channel channel)
{
    switch(format)
    {
        // Single planar formats have a single plane
        case Format::U8:
        case Format::U16:
        case Format::S16:
        case Format::U32:
        case Format::S32:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::F32:
        case Format::UV88:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
            return 0;
        // Multi planar formats
        case Format::NV12:
        case Format::NV21:
        {
            // Channel U and V share the same plane of format UV88
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                case Channel::V:
                    return 1;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::IYUV:
        case Format::YUV444:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 1;
                case Channel::V:
                    return 2;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        default:
            ARM_COMPUTE_ERROR("Not supported format");
            return 0;
    }
}

/** Return the channel index of a given channel given an input format.
 *
 * @param[in] format  Input format
 * @param[in] channel Input channel
 *
 * @return The channel index of the specific channel of the specific format
 */
inline int channel_idx_from_format(Format format, Channel channel)
{
    switch(format)
    {
        case Format::RGB888:
        {
            switch(channel)
            {
                case Channel::R:
                    return 0;
                case Channel::G:
                    return 1;
                case Channel::B:
                    return 2;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::RGBA8888:
        {
            switch(channel)
            {
                case Channel::R:
                    return 0;
                case Channel::G:
                    return 1;
                case Channel::B:
                    return 2;
                case Channel::A:
                    return 3;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::YUYV422:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 1;
                case Channel::V:
                    return 3;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::UYVY422:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 1;
                case Channel::U:
                    return 0;
                case Channel::V:
                    return 2;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::NV12:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 0;
                case Channel::V:
                    return 1;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::NV21:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 1;
                case Channel::V:
                    return 0;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::YUV444:
        case Format::IYUV:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 0;
                case Channel::V:
                    return 0;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        default:
            ARM_COMPUTE_ERROR("Not supported format");
            return 0;
    }
}

/** Return the number of planes for a given format
 *
 * @param[in] format Input format
 *
 * @return The number of planes for a given image format.
 */
inline size_t num_planes_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
        case Format::S16:
        case Format::U16:
        case Format::S32:
        case Format::U32:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::F32:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
            return 1;
        case Format::NV12:
        case Format::NV21:
            return 2;
        case Format::IYUV:
        case Format::YUV444:
            return 3;
        default:
            ARM_COMPUTE_ERROR("Not supported format");
            return 0;
    }
}

/** Return the number of channels for a given single-planar pixel format
 *
 * @param[in] format Input format
 *
 * @return The number of channels for a given image format.
 */
inline size_t num_channels_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
        case Format::U16:
        case Format::S16:
        case Format::U32:
        case Format::S32:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::F32:
            return 1;
        // Because the U and V channels are subsampled
        // these formats appear like having only 2 channels:
        case Format::YUYV422:
        case Format::UYVY422:
            return 2;
        case Format::UV88:
            return 2;
        case Format::RGB888:
            return 3;
        case Format::RGBA8888:
            return 4;
        //Doesn't make sense for planar formats:
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        case Format::YUV444:
        default:
            return 0;
    }
}

/** Return the promoted data type of a given data type.
 *
 * @note If promoted data type is not supported an error will be thrown
 *
 * @param[in] dt Data type to get the promoted type of.
 *
 * @return Promoted data type
 */
inline DataType get_promoted_data_type(DataType dt)
{
    switch(dt)
    {
        case DataType::U8:
            return DataType::U16;
        case DataType::S8:
            return DataType::S16;
        case DataType::U16:
            return DataType::U32;
        case DataType::S16:
            return DataType::S32;
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
        case DataType::BFLOAT16:
        case DataType::F16:
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            ARM_COMPUTE_ERROR("Unsupported data type promotions!");
        default:
            ARM_COMPUTE_ERROR("Undefined data type!");
    }
    return DataType::UNKNOWN;
}

/** Compute the mininum and maximum values a data type can take
 *
 * @param[in] dt Data type to get the min/max bounds of
 *
 * @return A tuple (min,max) with the minimum and maximum values respectively wrapped in PixelValue.
 */
inline std::tuple<PixelValue, PixelValue> get_min_max(DataType dt)
{
    PixelValue min{};
    PixelValue max{};
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<uint8_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<uint8_t>::max()));
            break;
        }
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<int8_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<int8_t>::max()));
            break;
        }
        case DataType::U16:
        case DataType::QASYMM16:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<uint16_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<uint16_t>::max()));
            break;
        }
        case DataType::S16:
        case DataType::QSYMM16:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<int16_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<int16_t>::max()));
            break;
        }
        case DataType::U32:
        {
            min = PixelValue(std::numeric_limits<uint32_t>::lowest());
            max = PixelValue(std::numeric_limits<uint32_t>::max());
            break;
        }
        case DataType::S32:
        {
            min = PixelValue(std::numeric_limits<int32_t>::lowest());
            max = PixelValue(std::numeric_limits<int32_t>::max());
            break;
        }
        case DataType::BFLOAT16:
        {
            min = PixelValue(bfloat16::lowest());
            max = PixelValue(bfloat16::max());
            break;
        }
        case DataType::F16:
        {
            min = PixelValue(std::numeric_limits<half>::lowest());
            max = PixelValue(std::numeric_limits<half>::max());
            break;
        }
        case DataType::F32:
        {
            min = PixelValue(std::numeric_limits<float>::lowest());
            max = PixelValue(std::numeric_limits<float>::max());
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Undefined data type!");
    }
    return std::make_tuple(min, max);
}

/** Return true if the given format has horizontal subsampling.
 *
 * @param[in] format Format to determine subsampling.
 *
 * @return True if the format can be subsampled horizontaly.
 */
inline bool has_format_horizontal_subsampling(Format format)
{
    return (format == Format::YUYV422 || format == Format::UYVY422 || format == Format::NV12 || format == Format::NV21 || format == Format::IYUV || format == Format::UV88) ? true : false;
}

/** Return true if the given format has vertical subsampling.
 *
 * @param[in] format Format to determine subsampling.
 *
 * @return True if the format can be subsampled verticaly.
 */
inline bool has_format_vertical_subsampling(Format format)
{
    return (format == Format::NV12 || format == Format::NV21 || format == Format::IYUV || format == Format::UV88) ? true : false;
}

/** Adjust tensor shape size if width or height are odd for a given multi-planar format. No modification is done for other formats.
 *
 * @note Adding here a few links discussing the issue of odd size and sharing the same solution:
 *       <a href="https://android.googlesource.com/platform/frameworks/base/+/refs/heads/master/graphics/java/android/graphics/YuvImage.java">Android Source</a>
 *       <a href="https://groups.google.com/a/webmproject.org/forum/#!topic/webm-discuss/LaCKpqiDTXM">WebM</a>
 *       <a href="https://bugs.chromium.org/p/libyuv/issues/detail?id=198&amp;can=1&amp;q=odd%20width">libYUV</a>
 *       <a href="https://sourceforge.net/p/raw-yuvplayer/bugs/1/">YUVPlayer</a> *
 *
 * @param[in, out] shape  Tensor shape of 2D size
 * @param[in]      format Format of the tensor
 *
 * @return The adjusted tensor shape.
 */
inline TensorShape adjust_odd_shape(const TensorShape &shape, Format format)
{
    TensorShape output{ shape };

    // Force width to be even for formats which require subsampling of the U and V channels
    if(has_format_horizontal_subsampling(format))
    {
        output.set(0, (output.x() + 1) & ~1U);
    }

    // Force height to be even for formats which require subsampling of the U and V channels
    if(has_format_vertical_subsampling(format))
    {
        output.set(1, (output.y() + 1) & ~1U);
    }

    return output;
}

/** Calculate subsampled shape for a given format and channel
 *
 * @param[in] shape   Shape of the tensor to calculate the extracted channel.
 * @param[in] format  Format of the tensor.
 * @param[in] channel Channel to create tensor shape to be extracted.
 *
 * @return The subsampled tensor shape.
 */
inline TensorShape calculate_subsampled_shape(const TensorShape &shape, Format format, Channel channel = Channel::UNKNOWN)
{
    TensorShape output{ shape };

    // Subsample shape only for U or V channel
    if(Channel::U == channel || Channel::V == channel || Channel::UNKNOWN == channel)
    {
        // Subsample width for the tensor shape when channel is U or V
        if(has_format_horizontal_subsampling(format))
        {
            output.set(0, output.x() / 2U);
        }

        // Subsample height for the tensor shape when channel is U or V
        if(has_format_vertical_subsampling(format))
        {
            output.set(1, output.y() / 2U);
        }
    }

    return output;
}

/** Permutes the given dimensions according the permutation vector
 *
 * @param[in,out] dimensions Dimensions to be permuted.
 * @param[in]     perm       Vector describing the permutation.
 *
 */
template <typename T>
inline void permute_strides(Dimensions<T> &dimensions, const PermutationVector &perm)
{
    const auto old_dim = utility::make_array<Dimensions<T>::num_max_dimensions>(dimensions.begin(), dimensions.end());
    for(unsigned int i = 0; i < perm.num_dimensions(); ++i)
    {
        T dimension_val = old_dim[i];
        dimensions.set(perm[i], dimension_val);
    }
}

/** Calculate padding requirements in case of SAME padding
 *
 * @param[in] input_shape   Input shape
 * @param[in] weights_shape Weights shape
 * @param[in] conv_info     Convolution information (containing strides)
 * @param[in] data_layout   (Optional) Data layout of the input and weights tensor
 * @param[in] dilation      (Optional) Dilation factor used in the convolution.
 * @param[in] rounding_type (Optional) Dimension rounding type when down-scaling.
 *
 * @return PadStrideInfo for SAME padding
 */
PadStrideInfo calculate_same_pad(TensorShape input_shape, TensorShape weights_shape, PadStrideInfo conv_info, DataLayout data_layout = DataLayout::NCHW, const Size2D &dilation = Size2D(1u, 1u),
                                 const DimensionRoundingType &rounding_type = DimensionRoundingType::FLOOR);

/** Returns expected width and height of the deconvolution's output tensor.
 *
 * @param[in] in_width        Width of input tensor (Number of columns)
 * @param[in] in_height       Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
std::pair<unsigned int, unsigned int> deconvolution_output_dimensions(unsigned int in_width, unsigned int in_height,
                                                                      unsigned int kernel_width, unsigned int kernel_height,
                                                                      const PadStrideInfo &pad_stride_info);

/** Returns expected width and height of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width           Width of input tensor (Number of columns)
 * @param[in] height          Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 * @param[in] dilation        (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
std::pair<unsigned int, unsigned int> scaled_dimensions(int width, int height,
                                                        int kernel_width, int kernel_height,
                                                        const PadStrideInfo &pad_stride_info,
                                                        const Size2D        &dilation = Size2D(1U, 1U));

/** Returns calculated width and height of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width           Width of input tensor (Number of columns)
 * @param[in] height          Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 *
 * @return A pair with the new width in the first position and the new height in the second, returned values can be < 1
 */
std::pair<int, int> scaled_dimensions_signed(int width, int height,
                                             int kernel_width, int kernel_height,
                                             const PadStrideInfo &pad_stride_info);

/** Returns calculated width, height and depth of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width         Width of input tensor
 * @param[in] height        Height of input tensor
 * @param[in] depth         Depth of input tensor
 * @param[in] kernel_width  Kernel width.
 * @param[in] kernel_height Kernel height.
 * @param[in] kernel_depth  Kernel depth.
 * @param[in] pool3d_info   Pad and stride and round information for 3d pooling
 *
 * @return A tuple with the new width in the first position, the new height in the second, and the new depth in the third.
 *         Returned values can be < 1
 */
std::tuple<int, int, int> scaled_3d_dimensions_signed(int width, int height, int depth,
                                                      int kernel_width, int kernel_height, int kernel_depth,
                                                      const Pooling3dLayerInfo &pool3d_info);

/** Check if the given reduction operation should be handled in a serial way.
 *
 * @param[in] op   Reduction operation to perform
 * @param[in] dt   Data type
 * @param[in] axis Axis along which to reduce
 *
 * @return True if the given reduction operation should be handled in a serial way.
 */
bool needs_serialized_reduction(ReductionOperation op, DataType dt, unsigned int axis);

/** Returns output quantization information for softmax layer
 *
 * @param[in] input_type The data type of the input tensor
 * @param[in] is_log     True for log softmax
 *
 * @return Quantization information for the output tensor
 */
QuantizationInfo get_softmax_output_quantization_info(DataType input_type, bool is_log);

/** Returns a pair of minimum and maximum values for a quantized activation
 *
 * @param[in] act_info  The information for activation
 * @param[in] data_type The used data type
 * @param[in] oq_info   The output quantization information
 *
 * @return The pair with minimum and maximum values
 */
std::pair<int32_t, int32_t> get_quantized_activation_min_max(ActivationLayerInfo act_info, DataType data_type, UniformQuantizationInfo oq_info);

/** Convert a tensor format into a string.
 *
 * @param[in] format @ref Format to be translated to string.
 *
 * @return The string describing the format.
 */
const std::string &string_from_format(Format format);

/** Convert a channel identity into a string.
 *
 * @param[in] channel @ref Channel to be translated to string.
 *
 * @return The string describing the channel.
 */
const std::string &string_from_channel(Channel channel);
/** Convert a data layout identity into a string.
 *
 * @param[in] dl @ref DataLayout to be translated to string.
 *
 * @return The string describing the data layout.
 */
const std::string &string_from_data_layout(DataLayout dl);
/** Convert a data type identity into a string.
 *
 * @param[in] dt @ref DataType to be translated to string.
 *
 * @return The string describing the data type.
 */
const std::string &string_from_data_type(DataType dt);
/** Translates a given activation function to a string.
 *
 * @param[in] act @ref ActivationLayerInfo::ActivationFunction to be translated to string.
 *
 * @return The string describing the activation function.
 */
const std::string &string_from_activation_func(ActivationLayerInfo::ActivationFunction act);
/** Translates a given interpolation policy to a string.
 *
 * @param[in] policy @ref InterpolationPolicy to be translated to string.
 *
 * @return The string describing the interpolation policy.
 */
const std::string &string_from_interpolation_policy(InterpolationPolicy policy);
/** Translates a given border mode policy to a string.
 *
 * @param[in] border_mode @ref BorderMode to be translated to string.
 *
 * @return The string describing the border mode.
 */
const std::string &string_from_border_mode(BorderMode border_mode);
/** Translates a given normalization type to a string.
 *
 * @param[in] type @ref NormType to be translated to string.
 *
 * @return The string describing the normalization type.
 */
const std::string &string_from_norm_type(NormType type);
/** Translates a given pooling type to a string.
 *
 * @param[in] type @ref PoolingType to be translated to string.
 *
 * @return The string describing the pooling type.
 */
const std::string &string_from_pooling_type(PoolingType type);
/** Check if the pool region is entirely outside the input tensor
 *
 * @param[in] info @ref PoolingLayerInfo to be checked.
 *
 * @return True if the pool region is entirely outside the input tensor, False otherwise.
 */
bool is_pool_region_entirely_outside_input(const PoolingLayerInfo &info);
/** Check if the 3d pool region is entirely outside the input tensor
 *
 * @param[in] info @ref Pooling3dLayerInfo to be checked.
 *
 * @return True if the pool region is entirely outside the input tensor, False otherwise.
 */
bool is_pool_3d_region_entirely_outside_input(const Pooling3dLayerInfo &info);
/** Check if the 3D padding is symmetric i.e. padding in each opposite sides are euqal (left=right, top=bottom and front=back)
 *
 * @param[in] info @ref Padding3D input 3D padding object to check if it is symmetric
 *
 * @return True if padding is symmetric
 */
inline bool is_symmetric(const Padding3D& info)
{
    return ((info.left == info.right) && (info.top == info.bottom) && (info.front == info.back));
}
/** Translates a given GEMMLowp output stage to a string.
 *
 * @param[in] output_stage @ref GEMMLowpOutputStageInfo to be translated to string.
 *
 * @return The string describing the GEMMLowp output stage
 */
const std::string &string_from_gemmlowp_output_stage(GEMMLowpOutputStageType output_stage);
/** Convert a PixelValue to a string, represented through the specific data type
 *
 * @param[in] value     The PixelValue to convert
 * @param[in] data_type The type to be used to convert the @p value
 *
 * @return String representation of the PixelValue through the given data type.
 */
std::string string_from_pixel_value(const PixelValue &value, const DataType data_type);
/** Convert a string to DataType
 *
 * @param[in] name The name of the data type
 *
 * @return DataType
 */
DataType data_type_from_name(const std::string &name);
/** Stores padding information before configuring a kernel
 *
 * @param[in] infos list of tensor infos to store the padding info for
 *
 * @return An unordered map where each tensor info pointer is paired with its original padding info
 */
std::unordered_map<const ITensorInfo *, PaddingSize> get_padding_info(std::initializer_list<const ITensorInfo *> infos);
/** Stores padding information before configuring a kernel
 *
 * @param[in] tensors list of tensors to store the padding info for
 *
 * @return An unordered map where each tensor info pointer is paired with its original padding info
 */
std::unordered_map<const ITensorInfo *, PaddingSize> get_padding_info(std::initializer_list<const ITensor *> tensors);
/** Check if the previously stored padding info has changed after configuring a kernel
 *
 * @param[in] padding_map an unordered map where each tensor info pointer is paired with its original padding info
 *
 * @return true if any of the tensor infos has changed its paddings
 */
bool has_padding_changed(const std::unordered_map<const ITensorInfo *, PaddingSize> &padding_map);

/** Input Stream operator for @ref DataType
 *
 * @param[in]  stream    Stream to parse
 * @param[out] data_type Output data type
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, DataType &data_type)
{
    std::string value;
    stream >> value;
    data_type = data_type_from_name(value);
    return stream;
}
/** Lower a given string.
 *
 * @param[in] val Given string to lower.
 *
 * @return The lowered string
 */
std::string lower_string(const std::string &val);

/** Raise a given string to upper case
 *
 * @param[in] val Given string to lower.
 *
 * @return The upper case string
 */
std::string upper_string(const std::string &val);

/** Check if a given data type is of floating point type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of floating point type, else false.
 */
inline bool is_data_type_float(DataType dt)
{
    switch(dt)
    {
        case DataType::F16:
        case DataType::F32:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of quantized type
 *
 * @note Quantized is considered a super-set of fixed-point and asymmetric data types.
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of quantized type, else false.
 */
inline bool is_data_type_quantized(DataType dt)
{
    switch(dt)
    {
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of asymmetric quantized type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of asymmetric quantized type, else false.
 */
inline bool is_data_type_quantized_asymmetric(DataType dt)
{
    switch(dt)
    {
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QASYMM16:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of asymmetric quantized signed type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of asymmetric quantized signed type, else false.
 */
inline bool is_data_type_quantized_asymmetric_signed(DataType dt)
{
    switch(dt)
    {
        case DataType::QASYMM8_SIGNED:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of symmetric quantized type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of symmetric quantized type, else false.
 */
inline bool is_data_type_quantized_symmetric(DataType dt)
{
    switch(dt)
    {
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
        case DataType::QSYMM16:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of per channel type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of per channel type, else false.
 */
inline bool is_data_type_quantized_per_channel(DataType dt)
{
    switch(dt)
    {
        case DataType::QSYMM8_PER_CHANNEL:
            return true;
        default:
            return false;
    }
}

/** Create a string with the float in full precision.
 *
 * @param val Floating point value
 *
 * @return String with the floating point value.
 */
inline std::string float_to_string_with_full_precision(float val)
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<float>::max_digits10);
    ss << val;

    if(val != static_cast<int>(val))
    {
        ss << "f";
    }

    return ss.str();
}

/** Returns the number of elements required to go from start to end with the wanted step
 *
 * @param[in] start start value
 * @param[in] end   end value
 * @param[in] step  step value between each number in the wanted sequence
 *
 * @return number of elements to go from start value to end value using the wanted step
 */
inline size_t num_of_elements_in_range(const float start, const float end, const float step)
{
    ARM_COMPUTE_ERROR_ON_MSG(step == 0, "Range Step cannot be 0");
    return size_t(std::ceil((end - start) / step));
}

/** Returns true if the value can be represented by the given data type
 *
 * @param[in] val   value to be checked
 * @param[in] dt    data type that is checked
 * @param[in] qinfo (Optional) quantization info if the data type is QASYMM8
 *
 * @return true if the data type can hold the value.
 */
template <typename T>
bool check_value_range(T val, DataType dt, QuantizationInfo qinfo = QuantizationInfo())
{
    switch(dt)
    {
        case DataType::U8:
        {
            const auto val_u8 = static_cast<uint8_t>(val);
            return ((val_u8 == val) && val >= std::numeric_limits<uint8_t>::lowest() && val <= std::numeric_limits<uint8_t>::max());
        }
        case DataType::QASYMM8:
        {
            double min = static_cast<double>(dequantize_qasymm8(0, qinfo));
            double max = static_cast<double>(dequantize_qasymm8(std::numeric_limits<uint8_t>::max(), qinfo));
            return ((double)val >= min && (double)val <= max);
        }
        case DataType::S8:
        {
            const auto val_s8 = static_cast<int8_t>(val);
            return ((val_s8 == val) && val >= std::numeric_limits<int8_t>::lowest() && val <= std::numeric_limits<int8_t>::max());
        }
        case DataType::U16:
        {
            const auto val_u16 = static_cast<uint16_t>(val);
            return ((val_u16 == val) && val >= std::numeric_limits<uint16_t>::lowest() && val <= std::numeric_limits<uint16_t>::max());
        }
        case DataType::S16:
        {
            const auto val_s16 = static_cast<int16_t>(val);
            return ((val_s16 == val) && val >= std::numeric_limits<int16_t>::lowest() && val <= std::numeric_limits<int16_t>::max());
        }
        case DataType::U32:
        {
            const auto val_d64 = static_cast<double>(val);
            const auto val_u32 = static_cast<uint32_t>(val);
            return ((val_u32 == val_d64) && val_d64 >= std::numeric_limits<uint32_t>::lowest() && val_d64 <= std::numeric_limits<uint32_t>::max());
        }
        case DataType::S32:
        {
            const auto val_d64 = static_cast<double>(val);
            const auto val_s32 = static_cast<int32_t>(val);
            return ((val_s32 == val_d64) && val_d64 >= std::numeric_limits<int32_t>::lowest() && val_d64 <= std::numeric_limits<int32_t>::max());
        }
        case DataType::BFLOAT16:
            return (val >= bfloat16::lowest() && val <= bfloat16::max());
        case DataType::F16:
            return (val >= std::numeric_limits<half>::lowest() && val <= std::numeric_limits<half>::max());
        case DataType::F32:
            return (val >= std::numeric_limits<float>::lowest() && val <= std::numeric_limits<float>::max());
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            return false;
    }
}

/** Returns the adjusted vector size in case it is less than the input's first dimension, getting rounded down to its closest valid vector size
 *
 * @param[in] vec_size vector size to be adjusted
 * @param[in] dim0     size of the first dimension
 *
 * @return the number of element processed along the X axis per thread
 */
inline unsigned int adjust_vec_size(unsigned int vec_size, size_t dim0)
{
    ARM_COMPUTE_ERROR_ON(vec_size > 16);

    if((vec_size >= dim0) && (dim0 == 3))
    {
        return dim0;
    }

    while(vec_size > dim0)
    {
        vec_size >>= 1;
    }

    return vec_size;
}

/** Returns the suffix string of CPU kernel implementation names based on the given data type
 *
 * @param[in] data_type The data type the CPU kernel implemetation uses
 *
 * @return the suffix string of CPU kernel implementations
 */
inline std::string cpu_impl_dt(const DataType &data_type)
{
    std::string ret = "";

    switch(data_type)
    {
        case DataType::F32:
            ret = "fp32";
            break;
        case DataType::F16:
            ret = "fp16";
            break;
        case DataType::U8:
            ret = "u8";
            break;
        case DataType::S16:
            ret = "s16";
            break;
        case DataType::S32:
            ret = "s32";
            break;
        case DataType::QASYMM8:
            ret = "qu8";
            break;
        case DataType::QASYMM8_SIGNED:
            ret = "qs8";
            break;
        case DataType::QSYMM16:
            ret = "qs16";
            break;
        case DataType::QSYMM8_PER_CHANNEL:
            ret = "qp8";
            break;
        case DataType::BFLOAT16:
            ret = "bf16";
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported.");
    }

    return ret;
}

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
/** Print consecutive elements to an output stream.
 *
 * @param[out] s             Output stream to print the elements to.
 * @param[in]  ptr           Pointer to print the elements from.
 * @param[in]  n             Number of elements to print.
 * @param[in]  stream_width  (Optional) Width of the stream. If set to 0 the element's width is used. Defaults to 0.
 * @param[in]  element_delim (Optional) Delimeter among the consecutive elements. Defaults to space delimeter
 */
template <typename T>
void print_consecutive_elements_impl(std::ostream &s, const T *ptr, unsigned int n, int stream_width = 0, const std::string &element_delim = " ")
{
    using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;
    std::ios stream_status(nullptr);
    stream_status.copyfmt(s);

    for(unsigned int i = 0; i < n; ++i)
    {
        // Set stream width as it is not a "sticky" stream manipulator
        if(stream_width != 0)
        {
            s.width(stream_width);
        }

        if(std::is_same<typename std::decay<T>::type, half>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
            s << std::right << static_cast<T>(ptr[i]) << element_delim;
        }
        else if(std::is_same<typename std::decay<T>::type, bfloat16>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<bfloat16> returns false and then the print_type becomes int.
            s << std::right << float(ptr[i]) << element_delim;
        }
        else
        {
            s << std::right << static_cast<print_type>(ptr[i]) << element_delim;
        }
    }

    // Restore output stream flags
    s.copyfmt(stream_status);
}

/** Identify the maximum width of n consecutive elements.
 *
 * @param[in] s   The output stream which will be used to print the elements. Used to extract the stream format.
 * @param[in] ptr Pointer to the elements.
 * @param[in] n   Number of elements.
 *
 * @return The maximum width of the elements.
 */
template <typename T>
int max_consecutive_elements_display_width_impl(std::ostream &s, const T *ptr, unsigned int n)
{
    using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;

    int max_width = -1;
    for(unsigned int i = 0; i < n; ++i)
    {
        std::stringstream ss;
        ss.copyfmt(s);

        if(std::is_same<typename std::decay<T>::type, half>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
            ss << static_cast<T>(ptr[i]);
        }
        else if(std::is_same<typename std::decay<T>::type, bfloat16>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<bfloat> returns false and then the print_type becomes int.
            ss << float(ptr[i]);
        }
        else
        {
            ss << static_cast<print_type>(ptr[i]);
        }

        max_width = std::max<int>(max_width, ss.str().size());
    }
    return max_width;
}

/** Print consecutive elements to an output stream.
 *
 * @param[out] s             Output stream to print the elements to.
 * @param[in]  dt            Data type of the elements
 * @param[in]  ptr           Pointer to print the elements from.
 * @param[in]  n             Number of elements to print.
 * @param[in]  stream_width  (Optional) Width of the stream. If set to 0 the element's width is used. Defaults to 0.
 * @param[in]  element_delim (Optional) Delimeter among the consecutive elements. Defaults to space delimeter
 */
void print_consecutive_elements(std::ostream &s, DataType dt, const uint8_t *ptr, unsigned int n, int stream_width, const std::string &element_delim = " ");

/** Identify the maximum width of n consecutive elements.
 *
 * @param[in] s   Output stream to print the elements to.
 * @param[in] dt  Data type of the elements
 * @param[in] ptr Pointer to print the elements from.
 * @param[in] n   Number of elements to print.
 *
 * @return The maximum width of the elements.
 */
int max_consecutive_elements_display_width(std::ostream &s, DataType dt, const uint8_t *ptr, unsigned int n);
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */
}
#endif /*ARM_COMPUTE_UTILS_H */
