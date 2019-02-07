/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_UTILS_H__
#define __ARM_COMPUTE_UTILS_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Rounding.h"
#include "arm_compute/core/Types.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace arm_compute
{
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

/** Returns the arm_compute library build information
 *
 * Contains the version number and the build options used to build the library
 *
 * @return The arm_compute library build information
 */
std::string build_information();

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
        case DataType::QASYMM8:
            return 1;
        case DataType::U16:
        case DataType::S16:
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
        case DataType::QASYMM8:
            return 1;
        case DataType::U16:
        case DataType::S16:
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
        case DataType::QASYMM8:
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

/** Separate a 2D convolution into two 1D convolutions
 *
 * @param[in]  conv     2D convolution
 * @param[out] conv_col 1D vertical convolution
 * @param[out] conv_row 1D horizontal convolution
 * @param[in]  size     Size of the 2D convolution
 *
 * @return true if the separation was successful
 */
inline bool separate_matrix(const int16_t *conv, int16_t *conv_col, int16_t *conv_row, uint8_t size)
{
    int32_t min_col     = -1;
    int16_t min_col_val = -1;

    for(int32_t i = 0; i < size; ++i)
    {
        if(conv[i] != 0 && (min_col < 0 || abs(min_col_val) > abs(conv[i])))
        {
            min_col     = i;
            min_col_val = conv[i];
        }
    }

    if(min_col < 0)
    {
        return false;
    }

    for(uint32_t j = 0; j < size; ++j)
    {
        conv_col[j] = conv[min_col + j * size];
    }

    for(uint32_t i = 0; i < size; i++)
    {
        if(static_cast<int>(i) == min_col)
        {
            conv_row[i] = 1;
        }
        else
        {
            int16_t coeff = conv[i] / conv[min_col];

            for(uint32_t j = 1; j < size; ++j)
            {
                if(conv[i + j * size] != (conv_col[j] * coeff))
                {
                    return false;
                }
            }

            conv_row[i] = coeff;
        }
    }

    return true;
}

/** Calculate the scale of the given square matrix
 *
 * The scale is the absolute value of the sum of all the coefficients in the matrix.
 *
 * @note If the coefficients add up to 0 then the scale is set to 1.
 *
 * @param[in] matrix      Matrix coefficients
 * @param[in] matrix_size Number of elements per side of the square matrix. (Number of coefficients = matrix_size * matrix_size).
 *
 * @return The absolute value of the sum of the coefficients if they don't add up to 0, otherwise 1.
 */
inline uint32_t calculate_matrix_scale(const int16_t *matrix, unsigned int matrix_size)
{
    const size_t size = matrix_size * matrix_size;

    return std::max(1, std::abs(std::accumulate(matrix, matrix + size, 0)));
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
        output.set(0, output.x() & ~1U);
    }

    // Force height to be even for formats which require subsampling of the U and V channels
    if(has_format_vertical_subsampling(format))
    {
        output.set(1, output.y() & ~1U);
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

/** Calculate accurary required by the horizontal and vertical convolution computations
 *
 * @param[in] conv_col Pointer to the vertical vector of the separated convolution filter
 * @param[in] conv_row Pointer to the horizontal vector of the convolution filter
 * @param[in] size     Number of elements per vector of the separated matrix
 *
 * @return The return type is a pair. The first element of the pair is the biggest data type needed for the first stage. The second
 * element of the pair is the biggest data type needed for the second stage.
 */
inline std::pair<DataType, DataType> data_type_for_convolution(const int16_t *conv_col, const int16_t *conv_row, size_t size)
{
    DataType first_stage  = DataType::UNKNOWN;
    DataType second_stage = DataType::UNKNOWN;

    auto gez = [](const int16_t &v)
    {
        return v >= 0;
    };

    auto accu_neg = [](const int &first, const int &second)
    {
        return first + (second < 0 ? second : 0);
    };

    auto accu_pos = [](const int &first, const int &second)
    {
        return first + (second > 0 ? second : 0);
    };

    const bool only_positive_coefficients = std::all_of(conv_row, conv_row + size, gez) && std::all_of(conv_col, conv_col + size, gez);

    if(only_positive_coefficients)
    {
        const int max_row_value = std::accumulate(conv_row, conv_row + size, 0) * UINT8_MAX;
        const int max_value     = std::accumulate(conv_col, conv_col + size, 0) * max_row_value;

        first_stage = (max_row_value <= UINT16_MAX) ? DataType::U16 : DataType::S32;

        second_stage = (max_value <= UINT16_MAX) ? DataType::U16 : DataType::S32;
    }
    else
    {
        const int min_row_value  = std::accumulate(conv_row, conv_row + size, 0, accu_neg) * UINT8_MAX;
        const int max_row_value  = std::accumulate(conv_row, conv_row + size, 0, accu_pos) * UINT8_MAX;
        const int neg_coeffs_sum = std::accumulate(conv_col, conv_col + size, 0, accu_neg);
        const int pos_coeffs_sum = std::accumulate(conv_col, conv_col + size, 0, accu_pos);
        const int min_value      = neg_coeffs_sum * max_row_value + pos_coeffs_sum * min_row_value;
        const int max_value      = neg_coeffs_sum * min_row_value + pos_coeffs_sum * max_row_value;

        first_stage = ((INT16_MIN <= min_row_value) && (max_row_value <= INT16_MAX)) ? DataType::S16 : DataType::S32;

        second_stage = ((INT16_MIN <= min_value) && (max_value <= INT16_MAX)) ? DataType::S16 : DataType::S32;
    }

    return std::make_pair(first_stage, second_stage);
}

/** Calculate the accuracy required by the squared convolution calculation.
 *
 *
 * @param[in] conv Pointer to the squared convolution matrix
 * @param[in] size The total size of the convolution matrix
 *
 * @return The return is the biggest data type needed to do the convolution
 */
inline DataType data_type_for_convolution_matrix(const int16_t *conv, size_t size)
{
    auto gez = [](const int16_t v)
    {
        return v >= 0;
    };

    const bool only_positive_coefficients = std::all_of(conv, conv + size, gez);

    if(only_positive_coefficients)
    {
        const int max_conv_value = std::accumulate(conv, conv + size, 0) * UINT8_MAX;
        if(max_conv_value <= UINT16_MAX)
        {
            return DataType::U16;
        }
        else
        {
            return DataType::S32;
        }
    }
    else
    {
        const int min_value = std::accumulate(conv, conv + size, 0, [](int a, int b)
        {
            return b < 0 ? a + b : a;
        })
        * UINT8_MAX;

        const int max_value = std::accumulate(conv, conv + size, 0, [](int a, int b)
        {
            return b > 0 ? a + b : a;
        })
        * UINT8_MAX;

        if((INT16_MIN <= min_value) && (INT16_MAX >= max_value))
        {
            return DataType::S16;
        }
        else
        {
            return DataType::S32;
        }
    }
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
 *
 * @return PadStrideInfo for SAME padding
 */
PadStrideInfo calculate_same_pad(TensorShape input_shape, TensorShape weights_shape, PadStrideInfo conv_info, DataLayout data_layout = DataLayout::NCHW);

/** Returns expected width and height of the deconvolution's output tensor.
 *
 * @param[in] in_width      Width of input tensor (Number of columns)
 * @param[in] in_height     Height of input tensor (Number of rows)
 * @param[in] kernel_width  Kernel width.
 * @param[in] kernel_height Kernel height.
 * @param[in] padx          X axis padding.
 * @param[in] pady          Y axis padding.
 * @param[in] stride_x      X axis input stride.
 * @param[in] stride_y      Y axis input stride.
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
const std::pair<unsigned int, unsigned int> deconvolution_output_dimensions(unsigned int in_width, unsigned int in_height,
                                                                            unsigned int kernel_width, unsigned int kernel_height,
                                                                            unsigned int padx, unsigned int pady,
                                                                            unsigned int stride_x, unsigned int stride_y);

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
const std::pair<unsigned int, unsigned int> scaled_dimensions(unsigned int width, unsigned int height,
                                                              unsigned int kernel_width, unsigned int kernel_height,
                                                              const PadStrideInfo &pad_stride_info,
                                                              const Size2D        &dilation = Size2D(1U, 1U));

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
/** Convert a matrix pattern into a string.
 *
 * @param[in] pattern @ref MatrixPattern to be translated to string.
 *
 * @return The string describing the matrix pattern.
 */
const std::string &string_from_matrix_pattern(MatrixPattern pattern);
/** Translates a given activation function to a string.
 *
 * @param[in] act @ref ActivationLayerInfo::ActivationFunction to be translated to string.
 *
 * @return The string describing the activation function.
 */
const std::string &string_from_activation_func(ActivationLayerInfo::ActivationFunction act);
/** Translates a given non linear function to a string.
 *
 * @param[in] function @ref NonLinearFilterFunction to be translated to string.
 *
 * @return The string describing the non linear function.
 */
const std::string &string_from_non_linear_filter_function(NonLinearFilterFunction function);
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
/** Lower a given string.
 *
 * @param[in] val Given string to lower.
 *
 * @return The lowered string
 */
std::string lower_string(const std::string &val);

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
        case DataType::QASYMM8:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of asymmetric quantized type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of symmetric quantized type, else false.
 */
inline bool is_data_type_quantized_asymmetric(DataType dt)
{
    switch(dt)
    {
        case DataType::QASYMM8:
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
 * @param[in] val        value to be checked
 * @param[in] dt         data type that is checked
 * @param[in] quant_info quantization info if the data type is QASYMM8
 *
 * @return true if the data type can hold the value.
 */
template <typename T>
bool check_value_range(T val, DataType dt, QuantizationInfo quant_info = QuantizationInfo())
{
    switch(dt)
    {
        case DataType::U8:
            return ((static_cast<uint8_t>(val) == val) && val >= std::numeric_limits<uint8_t>::lowest() && val <= std::numeric_limits<uint8_t>::max());
        case DataType::QASYMM8:
        {
            double min = static_cast<double>(quant_info.dequantize(0));
            double max = static_cast<double>(quant_info.dequantize(std::numeric_limits<uint8_t>::max()));
            return ((double)val >= min && (double)val <= max);
        }
        case DataType::S8:
            return ((static_cast<int8_t>(val) == val) && val >= std::numeric_limits<int8_t>::lowest() && val <= std::numeric_limits<int8_t>::max());
        case DataType::U16:
            return ((static_cast<uint16_t>(val) == val) && val >= std::numeric_limits<uint16_t>::lowest() && val <= std::numeric_limits<uint16_t>::max());
        case DataType::S16:
            return ((static_cast<int16_t>(val) == val) && val >= std::numeric_limits<int16_t>::lowest() && val <= std::numeric_limits<int16_t>::max());
        case DataType::U32:
            return ((static_cast<uint32_t>(val) == val) && val >= std::numeric_limits<uint32_t>::lowest() && val <= std::numeric_limits<uint32_t>::max());
        case DataType::S32:
            return ((static_cast<int32_t>(val) == val) && val >= std::numeric_limits<int32_t>::lowest() && val <= std::numeric_limits<int32_t>::max());
        case DataType::U64:
            return (val >= std::numeric_limits<uint64_t>::lowest() && val <= std::numeric_limits<uint64_t>::max());
        case DataType::S64:
            return (val >= std::numeric_limits<int64_t>::lowest() && val <= std::numeric_limits<int64_t>::max());
        case DataType::F16:
            return (val >= std::numeric_limits<half>::lowest() && val <= std::numeric_limits<half>::max());
        case DataType::F32:
            return (val >= std::numeric_limits<float>::lowest() && val <= std::numeric_limits<float>::max());
        case DataType::F64:
            return (val >= std::numeric_limits<double>::lowest() && val <= std::numeric_limits<double>::max());
        case DataType::SIZET:
            return ((static_cast<size_t>(val) == val) && val >= std::numeric_limits<size_t>::lowest() && val <= std::numeric_limits<size_t>::max());
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            return false;
    }
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
        else
        {
            s << std::right << static_cast<print_type>(ptr[i]) << element_delim;
        }
    }
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
#endif /*__ARM_COMPUTE_UTILS_H__ */
