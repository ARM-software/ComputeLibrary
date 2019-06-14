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
#ifndef __ARM_COMPUTE_TEST_UTILS_H__
#define __ARM_COMPUTE_TEST_UTILS_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#endif /* ARM_COMPUTE_CL */

#ifdef ARM_COMPUTE_GC
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#endif /* ARM_COMPUTE_GC */

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace arm_compute
{
#ifdef ARM_COMPUTE_CL
class CLTensor;
#endif /* ARM_COMPUTE_CL */
namespace test
{
/** Round floating-point value with half value rounding to positive infinity.
 *
 * @param[in] value floating-point value to be rounded.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T round_half_up(T value)
{
    return std::floor(value + 0.5f);
}

/** Round floating-point value with half value rounding to nearest even.
 *
 * @param[in] value   floating-point value to be rounded.
 * @param[in] epsilon precision.
 *
 * @return Floating-point value of rounded @p value.
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline T round_half_even(T value, T epsilon = std::numeric_limits<T>::epsilon())
{
    T positive_value = std::abs(value);
    T ipart          = 0;
    std::modf(positive_value, &ipart);
    // If 'value' is exactly halfway between two integers
    if(std::abs(positive_value - (ipart + 0.5f)) < epsilon)
    {
        // If 'ipart' is even then return 'ipart'
        if(std::fmod(ipart, 2.f) < epsilon)
        {
            return support::cpp11::copysign(ipart, value);
        }
        // Else return the nearest even integer
        return support::cpp11::copysign(std::ceil(ipart + 0.5f), value);
    }
    // Otherwise use the usual round to closest
    return support::cpp11::copysign(support::cpp11::round(positive_value), value);
}

namespace traits
{
// *INDENT-OFF*
// clang-format off
/** Promote a type */
template <typename T> struct promote { };
/** Promote uint8_t to uint16_t */
template <> struct promote<uint8_t> { using type = uint16_t; /**< Promoted type */ };
/** Promote int8_t to int16_t */
template <> struct promote<int8_t> { using type = int16_t; /**< Promoted type */ };
/** Promote uint16_t to uint32_t */
template <> struct promote<uint16_t> { using type = uint32_t; /**< Promoted type */ };
/** Promote int16_t to int32_t */
template <> struct promote<int16_t> { using type = int32_t; /**< Promoted type */ };
/** Promote uint32_t to uint64_t */
template <> struct promote<uint32_t> { using type = uint64_t; /**< Promoted type */ };
/** Promote int32_t to int64_t */
template <> struct promote<int32_t> { using type = int64_t; /**< Promoted type */ };
/** Promote float to float */
template <> struct promote<float> { using type = float; /**< Promoted type */ };
/** Promote half to half */
template <> struct promote<half> { using type = half; /**< Promoted type */ };

/** Get promoted type */
template <typename T>
using promote_t = typename promote<T>::type;

template <typename T>
using make_signed_conditional_t = typename std::conditional<std::is_integral<T>::value, std::make_signed<T>, std::common_type<T>>::type;

template <typename T>
using make_unsigned_conditional_t = typename std::conditional<std::is_integral<T>::value, std::make_unsigned<T>, std::common_type<T>>::type;

// clang-format on
// *INDENT-ON*
}

/** Look up the format corresponding to a channel.
 *
 * @param[in] channel Channel type.
 *
 * @return Format that contains the given channel.
 */
inline Format get_format_for_channel(Channel channel)
{
    switch(channel)
    {
        case Channel::R:
        case Channel::G:
        case Channel::B:
            return Format::RGB888;
        default:
            throw std::runtime_error("Unsupported channel");
    }
}

/** Return the format of a channel.
 *
 * @param[in] channel Channel type.
 *
 * @return Format of the given channel.
 */
inline Format get_channel_format(Channel channel)
{
    switch(channel)
    {
        case Channel::R:
        case Channel::G:
        case Channel::B:
            return Format::U8;
        default:
            throw std::runtime_error("Unsupported channel");
    }
}

/** Base case of foldl.
 *
 * @return value.
 */
template <typename F, typename T>
inline T foldl(F &&, const T &value)
{
    return value;
}

/** Base case of foldl.
 *
 * @return func(value1, value2).
 */
template <typename F, typename T, typename U>
inline auto foldl(F &&func, T &&value1, U &&value2) -> decltype(func(value1, value2))
{
    return func(value1, value2);
}

/** Fold left.
 *
 * @param[in] func    Binary function to be called.
 * @param[in] initial Initial value.
 * @param[in] value   Argument passed to the function.
 * @param[in] values  Remaining arguments.
 */
template <typename F, typename I, typename T, typename... Vs>
inline I foldl(F &&func, I &&initial, T &&value, Vs &&... values)
{
    return foldl(std::forward<F>(func), func(std::forward<I>(initial), std::forward<T>(value)), std::forward<Vs>(values)...);
}

/** Create a valid region based on tensor shape, border mode and border size
 *
 * @param[in] a_shape          Shape used as size of the valid region.
 * @param[in] border_undefined (Optional) Boolean indicating if the border mode is undefined.
 * @param[in] border_size      (Optional) Border size used to specify the region to exclude.
 *
 * @return A valid region starting at (0, 0, ...) with size of @p shape if @p border_undefined is false; otherwise
 *  return A valid region starting at (@p border_size.left, @p border_size.top, ...) with reduced size of @p shape.
 */
inline ValidRegion shape_to_valid_region(const TensorShape &a_shape, bool border_undefined = false, BorderSize border_size = BorderSize(0))
{
    ValidRegion valid_region{ Coordinates(), a_shape };

    Coordinates &anchor = valid_region.anchor;
    TensorShape &shape  = valid_region.shape;

    if(border_undefined)
    {
        ARM_COMPUTE_ERROR_ON(shape.num_dimensions() < 2);

        anchor.set(0, border_size.left);
        anchor.set(1, border_size.top);

        const int valid_shape_x = std::max(0, static_cast<int>(shape.x()) - static_cast<int>(border_size.left) - static_cast<int>(border_size.right));
        const int valid_shape_y = std::max(0, static_cast<int>(shape.y()) - static_cast<int>(border_size.top) - static_cast<int>(border_size.bottom));

        shape.set(0, valid_shape_x);
        shape.set(1, valid_shape_y);
    }

    return valid_region;
}

/** Create a valid region for Gaussian Pyramid Half based on tensor shape and valid region at level "i - 1" and border mode
 *
 * @note The border size is 2 in case of Gaussian Pyramid Half
 *
 * @param[in] a_shape          Shape used at level "i - 1" of Gaussian Pyramid Half
 * @param[in] a_valid_region   Valid region used at level "i - 1" of Gaussian Pyramid Half
 * @param[in] border_undefined (Optional) Boolean indicating if the border mode is undefined.
 *
 *  return The valid region for the level "i" of Gaussian Pyramid Half
 */
inline ValidRegion shape_to_valid_region_gaussian_pyramid_half(const TensorShape &a_shape, const ValidRegion &a_valid_region, bool border_undefined = false)
{
    constexpr int border_size = 2;

    ValidRegion valid_region{ Coordinates(), a_shape };

    Coordinates &anchor = valid_region.anchor;
    TensorShape &shape  = valid_region.shape;

    // Compute tensor shape for level "i" of Gaussian Pyramid Half
    // dst_width  = (src_width + 1) * 0.5f
    // dst_height = (src_height + 1) * 0.5f
    shape.set(0, (a_shape[0] + 1) * 0.5f);
    shape.set(1, (a_shape[1] + 1) * 0.5f);

    if(border_undefined)
    {
        ARM_COMPUTE_ERROR_ON(shape.num_dimensions() < 2);

        // Compute the left and top invalid borders
        float invalid_border_left = static_cast<float>(a_valid_region.anchor.x() + border_size) / 2.0f;
        float invalid_border_top  = static_cast<float>(a_valid_region.anchor.y() + border_size) / 2.0f;

        // For the new anchor point we can have 2 cases:
        // 1) If the width/height of the tensor shape is odd, we have to take the ceil value of (a_valid_region.anchor.x() + border_size) / 2.0f or (a_valid_region.anchor.y() + border_size / 2.0f
        // 2) If the width/height of the tensor shape is even, we have to take the floor value of (a_valid_region.anchor.x() + border_size) / 2.0f or (a_valid_region.anchor.y() + border_size) / 2.0f
        // In this manner we should be able to propagate correctly the valid region along all levels of the pyramid
        invalid_border_left = (a_shape[0] % 2) ? std::ceil(invalid_border_left) : std::floor(invalid_border_left);
        invalid_border_top  = (a_shape[1] % 2) ? std::ceil(invalid_border_top) : std::floor(invalid_border_top);

        // Set the anchor point
        anchor.set(0, static_cast<int>(invalid_border_left));
        anchor.set(1, static_cast<int>(invalid_border_top));

        // Compute shape
        // Calculate the right and bottom invalid borders at the previous level of the pyramid
        const float prev_invalid_border_right  = static_cast<float>(a_shape[0] - (a_valid_region.anchor.x() + a_valid_region.shape[0]));
        const float prev_invalid_border_bottom = static_cast<float>(a_shape[1] - (a_valid_region.anchor.y() + a_valid_region.shape[1]));

        // Calculate the right and bottom invalid borders at the current level of the pyramid
        const float invalid_border_right  = std::ceil((prev_invalid_border_right + static_cast<float>(border_size)) / 2.0f);
        const float invalid_border_bottom = std::ceil((prev_invalid_border_bottom + static_cast<float>(border_size)) / 2.0f);

        const int valid_shape_x = std::max(0, static_cast<int>(shape.x()) - static_cast<int>(invalid_border_left) - static_cast<int>(invalid_border_right));
        const int valid_shape_y = std::max(0, static_cast<int>(shape.y()) - static_cast<int>(invalid_border_top) - static_cast<int>(invalid_border_bottom));

        shape.set(0, valid_shape_x);
        shape.set(1, valid_shape_y);
    }

    return valid_region;
}

/** Create a valid region for Laplacian Pyramid based on tensor shape and valid region at level "i - 1" and border mode
 *
 * @note The border size is 2 in case of Laplacian Pyramid
 *
 * @param[in] a_shape          Shape used at level "i - 1" of Laplacian Pyramid
 * @param[in] a_valid_region   Valid region used at level "i - 1" of Laplacian Pyramid
 * @param[in] border_undefined (Optional) Boolean indicating if the border mode is undefined.
 *
 *  return The valid region for the level "i" of Laplacian Pyramid
 */
inline ValidRegion shape_to_valid_region_laplacian_pyramid(const TensorShape &a_shape, const ValidRegion &a_valid_region, bool border_undefined = false)
{
    ValidRegion valid_region = shape_to_valid_region_gaussian_pyramid_half(a_shape, a_valid_region, border_undefined);

    if(border_undefined)
    {
        const BorderSize gaussian5x5_border(2);

        auto border_left   = static_cast<int>(gaussian5x5_border.left);
        auto border_right  = static_cast<int>(gaussian5x5_border.right);
        auto border_top    = static_cast<int>(gaussian5x5_border.top);
        auto border_bottom = static_cast<int>(gaussian5x5_border.bottom);

        valid_region.anchor.set(0, valid_region.anchor[0] + border_left);
        valid_region.anchor.set(1, valid_region.anchor[1] + border_top);
        valid_region.shape.set(0, std::max(0, static_cast<int>(valid_region.shape[0]) - border_right - border_left));
        valid_region.shape.set(1, std::max(0, static_cast<int>(valid_region.shape[1]) - border_top - border_bottom));
    }

    return valid_region;
}

/** Write the value after casting the pointer according to @p data_type.
 *
 * @warning The type of the value must match the specified data type.
 *
 * @param[out] ptr       Pointer to memory where the @p value will be written.
 * @param[in]  value     Value that will be written.
 * @param[in]  data_type Data type that will be written.
 */
template <typename T>
void store_value_with_data_type(void *ptr, T value, DataType data_type)
{
    switch(data_type)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            *reinterpret_cast<uint8_t *>(ptr) = value;
            break;
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            *reinterpret_cast<int8_t *>(ptr) = value;
            break;
        case DataType::U16:
            *reinterpret_cast<uint16_t *>(ptr) = value;
            break;
        case DataType::S16:
        case DataType::QSYMM16:
            *reinterpret_cast<int16_t *>(ptr) = value;
            break;
        case DataType::U32:
            *reinterpret_cast<uint32_t *>(ptr) = value;
            break;
        case DataType::S32:
            *reinterpret_cast<int32_t *>(ptr) = value;
            break;
        case DataType::U64:
            *reinterpret_cast<uint64_t *>(ptr) = value;
            break;
        case DataType::S64:
            *reinterpret_cast<int64_t *>(ptr) = value;
            break;
        case DataType::F16:
            *reinterpret_cast<half *>(ptr) = value;
            break;
        case DataType::F32:
            *reinterpret_cast<float *>(ptr) = value;
            break;
        case DataType::F64:
            *reinterpret_cast<double *>(ptr) = value;
            break;
        case DataType::SIZET:
            *reinterpret_cast<size_t *>(ptr) = value;
            break;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

/** Saturate a value of type T against the numeric limits of type U.
 *
 * @param[in] val Value to be saturated.
 *
 * @return saturated value.
 */
template <typename U, typename T>
T saturate_cast(T val)
{
    if(val > static_cast<T>(std::numeric_limits<U>::max()))
    {
        val = static_cast<T>(std::numeric_limits<U>::max());
    }
    if(val < static_cast<T>(std::numeric_limits<U>::lowest()))
    {
        val = static_cast<T>(std::numeric_limits<U>::lowest());
    }
    return val;
}

/** Find the signed promoted common type.
 */
template <typename... T>
struct common_promoted_signed_type
{
    /** Common type */
    using common_type = typename std::common_type<T...>::type;
    /** Promoted type */
    using promoted_type = traits::promote_t<common_type>;
    /** Intermediate type */
    using intermediate_type = typename traits::make_signed_conditional_t<promoted_type>::type;
};

/** Find the unsigned promoted common type.
 */
template <typename... T>
struct common_promoted_unsigned_type
{
    /** Common type */
    using common_type = typename std::common_type<T...>::type;
    /** Promoted type */
    using promoted_type = traits::promote_t<common_type>;
    /** Intermediate type */
    using intermediate_type = typename traits::make_unsigned_conditional_t<promoted_type>::type;
};

/** Convert a linear index into n-dimensional coordinates.
 *
 * @param[in] shape Shape of the n-dimensional tensor.
 * @param[in] index Linear index specifying the i-th element.
 *
 * @return n-dimensional coordinates.
 */
inline Coordinates index2coord(const TensorShape &shape, int index)
{
    int num_elements = shape.total_size();

    ARM_COMPUTE_ERROR_ON_MSG(index < 0 || index >= num_elements, "Index has to be in [0, num_elements]");
    ARM_COMPUTE_ERROR_ON_MSG(num_elements == 0, "Cannot create coordinate from empty shape");

    Coordinates coord{ 0 };

    for(int d = shape.num_dimensions() - 1; d >= 0; --d)
    {
        num_elements /= shape[d];
        coord.set(d, index / num_elements);
        index %= num_elements;
    }

    return coord;
}

/** Linearise the given coordinate.
 *
 * Transforms the given coordinate into a linear offset in terms of
 * elements.
 *
 * @param[in] shape Shape of the n-dimensional tensor.
 * @param[in] coord The to be converted coordinate.
 *
 * @return Linear offset to the element.
 */
inline int coord2index(const TensorShape &shape, const Coordinates &coord)
{
    ARM_COMPUTE_ERROR_ON_MSG(shape.total_size() == 0, "Cannot get index from empty shape");
    ARM_COMPUTE_ERROR_ON_MSG(coord.num_dimensions() == 0, "Cannot get index of empty coordinate");

    int index    = 0;
    int dim_size = 1;

    for(unsigned int i = 0; i < coord.num_dimensions(); ++i)
    {
        index += coord[i] * dim_size;
        dim_size *= shape[i];
    }

    return index;
}

/** Check if a coordinate is within a valid region */
inline bool is_in_valid_region(const ValidRegion &valid_region, Coordinates coord)
{
    for(size_t d = 0; d < Coordinates::num_max_dimensions; ++d)
    {
        if(coord[d] < valid_region.start(d) || coord[d] >= valid_region.end(d))
        {
            return false;
        }
    }

    return true;
}

/** Create and initialize a tensor of the given type.
 *
 * @param[in] shape             Tensor shape.
 * @param[in] data_type         Data type.
 * @param[in] num_channels      (Optional) Number of channels.
 * @param[in] quantization_info (Optional) Quantization info for asymmetric quantized types.
 * @param[in] data_layout       (Optional) Data layout. Default is NCHW.
 *
 * @return Initialized tensor of given type.
 */
template <typename T>
inline T create_tensor(const TensorShape &shape, DataType data_type, int num_channels = 1,
                       QuantizationInfo quantization_info = QuantizationInfo(), DataLayout data_layout = DataLayout::NCHW)
{
    T          tensor;
    TensorInfo info(shape, num_channels, data_type);
    info.set_quantization_info(quantization_info);
    info.set_data_layout(data_layout);
    tensor.allocator()->init(info);

    return tensor;
}

/** Create and initialize a tensor of the given type.
 *
 * @param[in] shape  Tensor shape.
 * @param[in] format Format type.
 *
 * @return Initialized tensor of given type.
 */
template <typename T>
inline T create_tensor(const TensorShape &shape, Format format)
{
    TensorInfo info(shape, format);

    T tensor;
    tensor.allocator()->init(info);

    return tensor;
}

/** Create and initialize a multi-image of the given type.
 *
 * @param[in] shape  Tensor shape.
 * @param[in] format Format type.
 *
 * @return Initialized tensor of given type.
 */
template <typename T>
inline T create_multi_image(const TensorShape &shape, Format format)
{
    T multi_image;
    multi_image.init(shape.x(), shape.y(), format);

    return multi_image;
}

/** Create and initialize a HOG (Histogram of Oriented Gradients) of the given type.
 *
 * @param[in] hog_info HOGInfo object
 *
 * @return Initialized HOG of given type.
 */
template <typename T>
inline T create_HOG(const HOGInfo &hog_info)
{
    T hog;
    hog.init(hog_info);

    return hog;
}

/** Create and initialize a Pyramid of the given type.
 *
 * @param[in] pyramid_info The PyramidInfo object.
 *
 * @return Initialized Pyramid of given type.
 */
template <typename T>
inline T create_pyramid(const PyramidInfo &pyramid_info)
{
    T pyramid;
    pyramid.init_auto_padding(pyramid_info);

    return pyramid;
}

/** Initialize a convolution matrix.
 *
 * @param[in, out] conv   The input convolution matrix.
 * @param[in]      width  The width of the convolution matrix.
 * @param[in]      height The height of the convolution matrix.
 * @param[in]      seed   The random seed to be used.
 */
inline void init_conv(int16_t *conv, unsigned int width, unsigned int height, std::random_device::result_type seed)
{
    std::mt19937                           gen(seed);
    std::uniform_int_distribution<int16_t> distribution_int16(-32768, 32767);

    for(unsigned int i = 0; i < width * height; ++i)
    {
        conv[i] = distribution_int16(gen);
    }
}

/** Initialize a separable convolution matrix.
 *
 * @param[in, out] conv   The input convolution matrix.
 * @param[in]      width  The width of the convolution matrix.
 * @param[in]      height The height of the convolution matrix.
 * @param[in]      seed   The random seed to be used.
 */
inline void init_separable_conv(int16_t *conv, unsigned int width, unsigned int height, std::random_device::result_type seed)
{
    std::mt19937 gen(seed);
    // Set it between -128 and 127 to ensure the matrix does not overflow
    std::uniform_int_distribution<int16_t> distribution_int16(-128, 127);

    int16_t conv_row[width];
    int16_t conv_col[height];

    conv_row[0] = conv_col[0] = 1;
    for(unsigned int i = 1; i < width; ++i)
    {
        conv_row[i] = distribution_int16(gen);
    }

    for(unsigned int i = 1; i < height; ++i)
    {
        conv_col[i] = distribution_int16(gen);
    }

    // Multiply two matrices
    for(unsigned int i = 0; i < width; ++i)
    {
        for(unsigned int j = 0; j < height; ++j)
        {
            conv[i * width + j] = conv_col[i] * conv_row[j];
        }
    }
}

/** Create a vector with a uniform distribution of floating point values across the specified range.
 *
 * @param[in] num_values The number of values to be created.
 * @param[in] min        The minimum value in distribution (inclusive).
 * @param[in] max        The maximum value in distribution (inclusive).
 * @param[in] seed       The random seed to be used.
 *
 * @return A vector that contains the requested number of random floating point values
 */
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
inline std::vector<T> generate_random_real(unsigned int num_values, T min, T max, std::random_device::result_type seed)
{
    std::vector<T>                    v(num_values);
    std::mt19937                      gen(seed);
    std::uniform_real_distribution<T> dist(min, max);

    for(unsigned int i = 0; i < num_values; ++i)
    {
        v.at(i) = dist(gen);
    }

    return v;
}

/** Create a vector of random keypoints for pyramid representation.
 *
 * @param[in] shape         The shape of the input tensor.
 * @param[in] num_keypoints The number of keypoints to be created.
 * @param[in] seed          The random seed to be used.
 * @param[in] num_levels    The number of pyramid levels.
 *
 * @return A vector that contains the requested number of random keypoints
 */
inline std::vector<KeyPoint> generate_random_keypoints(const TensorShape &shape, size_t num_keypoints, std::random_device::result_type seed, size_t num_levels = 1)
{
    std::vector<KeyPoint> keypoints;
    std::mt19937          gen(seed);

    // Calculate distribution bounds
    const auto min        = static_cast<int>(std::pow(2, num_levels));
    const auto max_width  = static_cast<int>(shape.x());
    const auto max_height = static_cast<int>(shape.y());

    ARM_COMPUTE_ERROR_ON(min > max_width || min > max_height);

    // Create distributions
    std::uniform_int_distribution<> dist_w(min, max_width);
    std::uniform_int_distribution<> dist_h(min, max_height);

    for(unsigned int i = 0; i < num_keypoints; i++)
    {
        KeyPoint keypoint;
        keypoint.x               = dist_w(gen);
        keypoint.y               = dist_h(gen);
        keypoint.tracking_status = 1;

        keypoints.push_back(keypoint);
    }

    return keypoints;
}

template <typename T, typename ArrayAccessor_T>
inline void fill_array(ArrayAccessor_T &&array, const std::vector<T> &v)
{
    array.resize(v.size());
    std::memcpy(array.buffer(), v.data(), v.size() * sizeof(T));
}

/** Obtain numpy type string from DataType.
 *
 * @param[in] data_type Data type.
 *
 * @return numpy type string.
 */
inline std::string get_typestring(DataType data_type)
{
    // Check endianness
    const unsigned int i = 1;
    const char        *c = reinterpret_cast<const char *>(&i);
    std::string        endianness;
    if(*c == 1)
    {
        endianness = std::string("<");
    }
    else
    {
        endianness = std::string(">");
    }
    const std::string no_endianness("|");

    switch(data_type)
    {
        case DataType::U8:
            return no_endianness + "u" + support::cpp11::to_string(sizeof(uint8_t));
        case DataType::S8:
            return no_endianness + "i" + support::cpp11::to_string(sizeof(int8_t));
        case DataType::U16:
            return endianness + "u" + support::cpp11::to_string(sizeof(uint16_t));
        case DataType::S16:
            return endianness + "i" + support::cpp11::to_string(sizeof(int16_t));
        case DataType::U32:
            return endianness + "u" + support::cpp11::to_string(sizeof(uint32_t));
        case DataType::S32:
            return endianness + "i" + support::cpp11::to_string(sizeof(int32_t));
        case DataType::U64:
            return endianness + "u" + support::cpp11::to_string(sizeof(uint64_t));
        case DataType::S64:
            return endianness + "i" + support::cpp11::to_string(sizeof(int64_t));
        case DataType::F32:
            return endianness + "f" + support::cpp11::to_string(sizeof(float));
        case DataType::F64:
            return endianness + "f" + support::cpp11::to_string(sizeof(double));
        case DataType::SIZET:
            return endianness + "u" + support::cpp11::to_string(sizeof(size_t));
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

/** Sync if necessary.
 */
template <typename TensorType>
inline void sync_if_necessary()
{
#ifdef ARM_COMPUTE_CL
    if(opencl_is_available() && std::is_same<typename std::decay<TensorType>::type, arm_compute::CLTensor>::value)
    {
        CLScheduler::get().sync();
    }
#endif /* ARM_COMPUTE_CL */
}

/** Sync tensor if necessary.
 *
 * @note: If the destination tensor not being used on OpenGL ES, GPU will optimize out the operation.
 *
 * @param[in] tensor Tensor to be sync.
 */
template <typename TensorType>
inline void sync_tensor_if_necessary(TensorType &tensor)
{
#ifdef ARM_COMPUTE_GC
    if(opengles31_is_available() && std::is_same<typename std::decay<TensorType>::type, arm_compute::GCTensor>::value)
    {
        // Force sync the tensor by calling map and unmap.
        IGCTensor &t = dynamic_cast<IGCTensor &>(tensor);
        t.map();
        t.unmap();
    }
#endif /* ARM_COMPUTE_GC */
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_UTILS_H__ */
