/*
 * Copyright (c) 2016, 2017, 2018 ARM Limited.
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
/** Computes the smallest number larger or equal to value that is a multiple of divisor. */
template <typename S, typename T>
inline auto ceil_to_multiple(S value, T divisor) -> decltype(((value + divisor - 1) / divisor) * divisor)
{
    ARM_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
    return ((value + divisor - 1) / divisor) * divisor;
}

/** Computes the largest number smaller or equal to value that is a multiple of divisor. */
template <typename S, typename T>
inline auto floor_to_multiple(S value, T divisor) -> decltype((value / divisor) * divisor)
{
    ARM_COMPUTE_ERROR_ON(value < 0 || divisor <= 0);
    return (value / divisor) * divisor;
}

/** Calculate the rounded up quotient of val / m. */
template <typename S, typename T>
constexpr auto DIV_CEIL(S val, T m) -> decltype((val + m - 1) / m)
{
    return (val + m - 1) / m;
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
        case DataType::QS8:
        case DataType::QASYMM8:
            return 1;
        case DataType::U16:
        case DataType::S16:
        case DataType::F16:
        case DataType::QS16:
            return 2;
        case DataType::F32:
        case DataType::U32:
        case DataType::S32:
        case DataType::QS32:
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
        case DataType::QS8:
        case DataType::QASYMM8:
            return 1;
        case DataType::U16:
        case DataType::S16:
        case DataType::QS16:
        case DataType::F16:
            return 2;
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
        case DataType::QS32:
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
        case Format::NV12:
        case Format::NV21:
        {
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
        case DataType::QS8:
            return DataType::QS16;
        case DataType::U16:
            return DataType::U32;
        case DataType::S16:
            return DataType::S32;
        case DataType::QS16:
            return DataType::QS32;
        case DataType::QASYMM8:
        case DataType::F16:
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
        case DataType::QS32:
            ARM_COMPUTE_ERROR("Unsupported data type promotions!");
        default:
            ARM_COMPUTE_ERROR("Undefined data type!");
    }
    return DataType::UNKNOWN;
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

/** Calculate the output shapes of the depth concatenate function.
 *
 * @param[in] inputs_vector The vector that stores all the pointers to input.
 *
 * @return the output shape
 */
template <typename T>
TensorShape calculate_depth_concatenate_shape(const std::vector<T *> &inputs_vector)
{
    TensorShape out_shape = inputs_vector[0]->info()->tensor_shape();

    size_t max_x = 0;
    size_t max_y = 0;
    size_t depth = 0;

    for(const auto &tensor : inputs_vector)
    {
        ARM_COMPUTE_ERROR_ON(tensor == nullptr);
        const TensorShape shape = tensor->info()->tensor_shape();
        max_x                   = std::max(shape.x(), max_x);
        max_y                   = std::max(shape.y(), max_y);
        depth += shape.z();
    }

    out_shape.set(0, max_x);
    out_shape.set(1, max_y);
    out_shape.set(2, depth);

    return out_shape;
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

/** Returns expected shape for the deconvolution output tensor.
 *
 * @param[in] out_dims widht and height of the output tensor, these values can be obtained with the function deconvolution_output_dimensions.
 * @param[in] input    Shape of the input tensor.
 * @param[in] weights  Shape of the weights tensor.
 *
 * @return Deconvolution output tensor shape.
 */
TensorShape deconvolution_output_shape(const std::pair<unsigned int, unsigned int> &out_dims, TensorShape input, TensorShape weights);

/** Returns expected width and height of the deconvolution's output tensor.
 *
 * @param[in] in_width           Width of input tensor (Number of columns)
 * @param[in] in_height          Height of input tensor (Number of rows)
 * @param[in] kernel_width       Kernel width.
 * @param[in] kernel_height      Kernel height.
 * @param[in] padx               X axis padding.
 * @param[in] pady               Y axis padding.
 * @param[in] inner_border_right The number of zeros added to right edge of the input.
 * @param[in] inner_border_top   The number of zeros added to top edge of the input.
 * @param[in] stride_x           X axis input stride.
 * @param[in] stride_y           Y axis input stride.
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
const std::pair<unsigned int, unsigned int> deconvolution_output_dimensions(unsigned int in_width, unsigned int in_height,
                                                                            unsigned int kernel_width, unsigned int kernel_height,
                                                                            unsigned int padx, unsigned int pady, unsigned int inner_border_right, unsigned int inner_border_top,
                                                                            unsigned int stride_x, unsigned int stride_y);

/** Returns expected width and height of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width           Width of input tensor (Number of columns)
 * @param[in] height          Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
const std::pair<unsigned int, unsigned int> scaled_dimensions(unsigned int width, unsigned int height,
                                                              unsigned int kernel_width, unsigned int kernel_height,
                                                              const PadStrideInfo &pad_stride_info);

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
        case DataType::QS8:
        case DataType::QASYMM8:
        case DataType::QS16:
        case DataType::QS32:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of fixed point type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of fixed point type, else false.
 */
inline bool is_data_type_fixed_point(DataType dt)
{
    switch(dt)
    {
        case DataType::QS8:
        case DataType::QS16:
        case DataType::QS32:
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
    ss.precision(std::numeric_limits<float>::digits10 + 1);
    ss << val;
    return ss.str();
}

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
}
#endif /*__ARM_COMPUTE_UTILS_H__ */
