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
#ifndef __ARM_COMPUTE_TEST_UTILS_H__
#define __ARM_COMPUTE_TEST_UTILS_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/HOGInfo.h"
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
template <typename T> struct promote { };
template <> struct promote<uint8_t> { using type = uint16_t; };
template <> struct promote<int8_t> { using type = int16_t; };
template <> struct promote<uint16_t> { using type = uint32_t; };
template <> struct promote<int16_t> { using type = int32_t; };
template <> struct promote<uint32_t> { using type = uint64_t; };
template <> struct promote<int32_t> { using type = int64_t; };
template <> struct promote<float> { using type = float; };
template <> struct promote<half> { using type = half; };


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
    shape.set(0, (shape[0] + 1) * 0.5f);
    shape.set(1, (shape[1] + 1) * 0.5f);

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
        invalid_border_left = (shape[0] % 2) ? std::ceil(invalid_border_left) : std::floor(invalid_border_left);
        invalid_border_top  = (shape[1] % 2) ? std::ceil(invalid_border_top) : std::floor(invalid_border_top);

        // Set the anchor point
        anchor.set(0, static_cast<int>(invalid_border_left));
        anchor.set(1, static_cast<int>(invalid_border_top));

        // Compute shape
        // Calculate the right and bottom invalid borders at the previous level of the pyramid
        const float prev_invalid_border_right  = static_cast<float>(shape[0] - (a_valid_region.anchor.x() + a_valid_region.shape[0]));
        const float prev_invalid_border_bottom = static_cast<float>(shape[1] - (a_valid_region.anchor.y() + a_valid_region.shape[1]));

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
        case DataType::QS8:
            *reinterpret_cast<int8_t *>(ptr) = value;
            break;
        case DataType::U16:
            *reinterpret_cast<uint16_t *>(ptr) = value;
            break;
        case DataType::S16:
        case DataType::QS16:
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
    using common_type       = typename std::common_type<T...>::type;
    using promoted_type     = traits::promote_t<common_type>;
    using intermediate_type = typename traits::make_signed_conditional_t<promoted_type>::type;
};

/** Find the unsigned promoted common type.
 */
template <typename... T>
struct common_promoted_unsigned_type
{
    using common_type       = typename std::common_type<T...>::type;
    using promoted_type     = traits::promote_t<common_type>;
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
 * @param[in] shape                Tensor shape.
 * @param[in] data_type            Data type.
 * @param[in] num_channels         (Optional) Number of channels.
 * @param[in] fixed_point_position (Optional) Number of fractional bits.
 * @param[in] quantization_info    (Optional) Quantization info for asymmetric quantized types.
 *
 * @return Initialized tensor of given type.
 */
template <typename T>
inline T create_tensor(const TensorShape &shape, DataType data_type, int num_channels = 1,
                       int fixed_point_position = 0, QuantizationInfo quantization_info = QuantizationInfo())
{
    T          tensor;
    TensorInfo info(shape, num_channels, data_type, fixed_point_position);
    info.set_quantization_info(quantization_info);
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

/** Create and initialize a HOG (Histogram of Oriented Gradients) of the given type.
 *
 * @param[in] cell_size             Cell size in pixels
 * @param[in] block_size            Block size in pixels. Must be a multiple of cell_size.
 * @param[in] detection_window_size Detection window size in pixels. Must be a multiple of block_size and block_stride.
 * @param[in] block_stride          Distance in pixels between 2 consecutive blocks along the x and y direction. Must be a multiple of cell size
 * @param[in] num_bins              Number of histogram bins for each cell
 * @param[in] normalization_type    (Optional) Normalization type to use for each block
 * @param[in] l2_hyst_threshold     (Optional) Threshold used for L2HYS_NORM normalization method
 * @param[in] phase_type            (Optional) Type of @ref PhaseType
 *
 * @return Initialized HOG of given type.
 */
template <typename T>
inline T create_HOG(const Size2D &cell_size, const Size2D &block_size, const Size2D &detection_window_size, const Size2D &block_stride, size_t num_bins,
                    HOGNormType normalization_type = HOGNormType::L2HYS_NORM, float l2_hyst_threshold = 0.2f, PhaseType phase_type = PhaseType::UNSIGNED)
{
    T       hog;
    HOGInfo hog_info(cell_size, block_size, block_size, block_stride, num_bins, normalization_type, l2_hyst_threshold, phase_type);
    hog.init(hog_info);

    return hog;
}

/** Create a vector of random ROIs.
 *
 * @param[in] shape     The shape of the input tensor.
 * @param[in] pool_info The ROI pooling information.
 * @param[in] num_rois  The number of ROIs to be created.
 * @param[in] seed      The random seed to be used.
 *
 * @return A vector that contains the requested number of random ROIs
 */
inline std::vector<ROI> generate_random_rois(const TensorShape &shape, const ROIPoolingLayerInfo &pool_info, unsigned int num_rois, std::random_device::result_type seed)
{
    ARM_COMPUTE_ERROR_ON((pool_info.pooled_width() < 4) || (pool_info.pooled_height() < 4));

    std::vector<ROI> rois;
    std::mt19937     gen(seed);
    const int        pool_width  = pool_info.pooled_width();
    const int        pool_height = pool_info.pooled_height();
    const float      roi_scale   = pool_info.spatial_scale();

    // Calculate distribution bounds
    const auto scaled_width  = static_cast<int>((shape.x() / roi_scale) / pool_width);
    const auto scaled_height = static_cast<int>((shape.y() / roi_scale) / pool_height);
    const auto min_width     = static_cast<int>(pool_width / roi_scale);
    const auto min_height    = static_cast<int>(pool_height / roi_scale);

    // Create distributions
    std::uniform_int_distribution<int> dist_batch(0, shape[3] - 1);
    std::uniform_int_distribution<int> dist_x(0, scaled_width);
    std::uniform_int_distribution<int> dist_y(0, scaled_height);
    std::uniform_int_distribution<int> dist_w(min_width, std::max(min_width, (pool_width - 2) * scaled_width));
    std::uniform_int_distribution<int> dist_h(min_height, std::max(min_height, (pool_height - 2) * scaled_height));

    for(unsigned int r = 0; r < num_rois; ++r)
    {
        ROI roi;
        roi.batch_idx   = dist_batch(gen);
        roi.rect.x      = dist_x(gen);
        roi.rect.y      = dist_y(gen);
        roi.rect.width  = dist_w(gen);
        roi.rect.height = dist_h(gen);
        rois.push_back(roi);
    }

    return rois;
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
