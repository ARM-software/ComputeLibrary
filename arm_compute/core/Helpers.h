/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_HELPERS_H__
#define __ARM_COMPUTE_HELPERS_H__

#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Steps.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/utility.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace arm_compute
{
class IKernel;
class ITensor;
class ITensorInfo;

template <typename T>
struct enable_bitwise_ops
{
    static constexpr bool value = false;
};

template <typename T>
typename std::enable_if<enable_bitwise_ops<T>::value, T>::type operator&(T lhs, T rhs)
{
    using underlying_type = typename std::underlying_type<T>::type;
    return static_cast<T>(static_cast<underlying_type>(lhs) & static_cast<underlying_type>(rhs));
}

namespace traits
{
/** Check if a type T is contained in a tuple Tuple of types */
template <typename T, typename Tuple>
struct is_contained;

template <typename T>
struct is_contained<T, std::tuple<>> : std::false_type
{
};

template <typename T, typename... Ts>
struct is_contained<T, std::tuple<T, Ts...>> : std::true_type
{
};

template <typename T, typename U, typename... Ts>
struct is_contained<T, std::tuple<U, Ts...>> : is_contained<T, std::tuple<Ts...>>
{
};
}

/** Computes bilinear interpolation using the pointer to the top-left pixel and the pixel's distance between
 * the real coordinates and the smallest following integer coordinates. Input must be in single channel format.
 *
 * @param[in] pixel_ptr Pointer to the top-left pixel value of a single channel input.
 * @param[in] stride    Stride to access the bottom-left and bottom-right pixel values
 * @param[in] dx        Pixel's distance between the X real coordinate and the smallest X following integer
 * @param[in] dy        Pixel's distance between the Y real coordinate and the smallest Y following integer
 *
 * @note dx and dy must be in the range [0, 1.0]
 *
 * @return The bilinear interpolated pixel value
 */
template <typename T>
inline T delta_bilinear_c1(const T *pixel_ptr, size_t stride, float dx, float dy)
{
    ARM_COMPUTE_ERROR_ON(pixel_ptr == nullptr);

    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;

    const T a00 = *pixel_ptr;
    const T a01 = *(pixel_ptr + 1);
    const T a10 = *(pixel_ptr + stride);
    const T a11 = *(pixel_ptr + stride + 1);

    const float w1 = dx1 * dy1;
    const float w2 = dx * dy1;
    const float w3 = dx1 * dy;
    const float w4 = dx * dy;

    return static_cast<T>(a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4);
}

/** Computes linear interpolation using the pointer to the top pixel and the pixel's distance between
 * the real coordinates and the smallest following integer coordinates. Input must be in single channel format.
 *
 * @param[in] pixel_ptr Pointer to the top pixel value of a single channel input.
 * @param[in] stride    Stride to access the bottom pixel value
 * @param[in] dy        Pixel's distance between the Y real coordinate and the smallest Y following integer
 *
 * @note dy must be in the range [0, 1.0]
 *
 * @return The linear interpolated pixel value
 */
template <typename T>
inline T delta_linear_c1_y(const T *pixel_ptr, size_t stride, float dy)
{
    ARM_COMPUTE_ERROR_ON(pixel_ptr == nullptr);

    const float dy1 = 1.0f - dy;

    const T a00 = *pixel_ptr;
    const T a10 = *(pixel_ptr + stride);

    const float w1 = dy1;
    const float w3 = dy;

    return static_cast<T>(a00 * w1 + a10 * w3);
}
/** Computes linear interpolation using the pointer to the left pixel and the pixel's distance between
 * the real coordinates and the smallest following integer coordinates. Input must be in single channel format.
 *
 * @param[in] pixel_ptr Pointer to the left pixel value of a single channel input.
 * @param[in] dx        Pixel's distance between the X real coordinate and the smallest X following integer
 *
 * @note dx must be in the range [0, 1.0]
 *
 * @return The linear interpolated pixel value
 */
template <typename T>
inline T delta_linear_c1_x(const T *pixel_ptr, float dx)
{
    ARM_COMPUTE_ERROR_ON(pixel_ptr == nullptr);

    const T a00 = *pixel_ptr;
    const T a01 = *(pixel_ptr + 1);

    const float dx1 = 1.0f - dx;

    const float w1 = dx1;
    const float w2 = dx;

    return static_cast<T>(a00 * w1 + a01 * w2);
}
/** Return the pixel at (x,y) using bilinear interpolation.
 *
 * @warning Only works if the iterator was created with an IImage
 *
 * @param[in] first_pixel_ptr Pointer to the first pixel of a single channel input.
 * @param[in] stride          Stride in bytes of the image;
 * @param[in] x               X position of the wanted pixel
 * @param[in] y               Y position of the wanted pixel
 *
 * @return The pixel at (x, y) using bilinear interpolation.
 */
template <typename T>
inline T pixel_bilinear_c1(const T *first_pixel_ptr, size_t stride, float x, float y)
{
    ARM_COMPUTE_ERROR_ON(first_pixel_ptr == nullptr);

    const int32_t xi = std::floor(x);
    const int32_t yi = std::floor(y);

    const float dx = x - xi;
    const float dy = y - yi;

    return delta_bilinear_c1(first_pixel_ptr + xi + yi * stride, stride, dx, dy);
}

/** Return the pixel at (x,y) using bilinear interpolation by clamping when out of borders. The image must be single channel input
 *
 * @warning Only works if the iterator was created with an IImage
 *
 * @param[in] first_pixel_ptr Pointer to the first pixel of a single channel image.
 * @param[in] stride          Stride in bytes of the image
 * @param[in] width           Width of the image
 * @param[in] height          Height of the image
 * @param[in] x               X position of the wanted pixel
 * @param[in] y               Y position of the wanted pixel
 *
 * @return The pixel at (x, y) using bilinear interpolation.
 */
template <typename T>
inline uint8_t pixel_bilinear_c1_clamp(const T *first_pixel_ptr, size_t stride, size_t width, size_t height, float x, float y)
{
    ARM_COMPUTE_ERROR_ON(first_pixel_ptr == nullptr);

    x = std::max(-1.f, std::min(x, static_cast<float>(width)));
    y = std::max(-1.f, std::min(y, static_cast<float>(height)));

    const float xi = std::floor(x);
    const float yi = std::floor(y);

    const float dx = x - xi;
    const float dy = y - yi;

    if(dx == 0.0f)
    {
        if(dy == 0.0f)
        {
            return static_cast<T>(first_pixel_ptr[static_cast<int32_t>(xi) + static_cast<int32_t>(yi) * stride]);
        }
        return delta_linear_c1_y(first_pixel_ptr + static_cast<int32_t>(xi) + static_cast<int32_t>(yi) * stride, stride, dy);
    }
    if(dy == 0.0f)
    {
        return delta_linear_c1_x(first_pixel_ptr + static_cast<int32_t>(xi) + static_cast<int32_t>(yi) * stride, dx);
    }
    return delta_bilinear_c1(first_pixel_ptr + static_cast<int32_t>(xi) + static_cast<int32_t>(yi) * stride, stride, dx, dy);
}

/** Return the pixel at (x,y) using area interpolation by clamping when out of borders. The image must be single channel U8
 *
 * @note The interpolation area depends on the width and height ration of the input and output images
 * @note Currently average of the contributing pixels is calculated
 *
 * @param[in] first_pixel_ptr Pointer to the first pixel of a single channel U8 image.
 * @param[in] stride          Stride in bytes of the image
 * @param[in] width           Width of the image
 * @param[in] height          Height of the image
 * @param[in] wr              Width ratio among the input image width and output image width.
 * @param[in] hr              Height ratio among the input image height and output image height.
 * @param[in] x               X position of the wanted pixel
 * @param[in] y               Y position of the wanted pixel
 *
 * @return The pixel at (x, y) using area interpolation.
 */
inline uint8_t pixel_area_c1u8_clamp(const uint8_t *first_pixel_ptr, size_t stride, size_t width, size_t height, float wr, float hr, int x, int y);

/** Performs clamping among a lower and upper value.
 *
 * @param[in] n     Value to clamp.
 * @param[in] lower Lower threshold.
 * @param[in] upper Upper threshold.
 *
 *  @return Clamped value.
 */
template <typename T>
inline T clamp(const T &n, const T &lower, const T &upper)
{
    return std::max(lower, std::min(n, upper));
}

/** Base case of for_each. Does nothing. */
template <typename F>
inline void for_each(F &&)
{
}

/** Call the function for each of the arguments
 *
 * @param[in] func Function to be called
 * @param[in] arg  Argument passed to the function
 * @param[in] args Remaining arguments
 */
template <typename F, typename T, typename... Ts>
inline void for_each(F &&func, T &&arg, Ts &&... args)
{
    func(arg);
    for_each(func, args...);
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
 * @return Function evaluation for value1 and value2
 */
template <typename F, typename T, typename U>
inline auto foldl(F &&func, T &&value1, U &&value2) -> decltype(func(value1, value2))
{
    return func(value1, value2);
}

/** Fold left.
 *
 * @param[in] func    Function to be called
 * @param[in] initial Initial value
 * @param[in] value   Argument passed to the function
 * @param[in] values  Remaining arguments
 */
template <typename F, typename I, typename T, typename... Vs>
inline I foldl(F &&func, I &&initial, T &&value, Vs &&... values)
{
    return foldl(std::forward<F>(func), func(std::forward<I>(initial), std::forward<T>(value)), std::forward<Vs>(values)...);
}

/** Iterator updated by @ref execute_window_loop for each window element */
class Iterator
{
public:
    /** Default constructor to create an empty iterator */
    constexpr Iterator();
    /** Create a container iterator for the metadata and allocation contained in the ITensor
     *
     * @param[in] tensor The tensor to associate to the iterator.
     * @param[in] window The window which will be used to iterate over the tensor.
     */
    Iterator(const ITensor *tensor, const Window &window);

    /** Increment the iterator along the specified dimension of the step value associated to the dimension.
     *
     * @warning It is the caller's responsibility to call increment(dimension+1) when reaching the end of a dimension, the iterator will not check for overflow.
     *
     * @note When incrementing a dimension 'n' the coordinates of all the dimensions in the range (0,n-1) are reset. For example if you iterate over a 2D image, everytime you change row (dimension 1), the iterator for the width (dimension 0) is reset to its start.
     *
     * @param[in] dimension Dimension to increment
     */
    void increment(size_t dimension);

    /** Return the offset in bytes from the first element to the current position of the iterator
     *
     * @return The current position of the iterator in bytes relative to the first element.
     */
    constexpr int offset() const;

    /** Return a pointer to the current pixel.
     *
     * @warning Only works if the iterator was created with an ITensor.
     *
     * @return equivalent to  buffer() + offset()
     */
    constexpr uint8_t *ptr() const;

    /** Move the iterator back to the beginning of the specified dimension.
     *
     * @param[in] dimension Dimension to reset
     */
    void reset(size_t dimension);

private:
    uint8_t *_ptr;

    class Dimension
    {
    public:
        constexpr Dimension()
            : _dim_start(0), _stride(0)
        {
        }

        int _dim_start;
        int _stride;
    };

    std::array<Dimension, Coordinates::num_max_dimensions> _dims;
};

/** Iterate through the passed window, automatically adjusting the iterators and calling the lambda_functino for each element.
 *  It passes the x and y positions to the lambda_function for each iteration
 *
 * @param[in]     w               Window to iterate through.
 * @param[in]     lambda_function The function of type void(function)( const Coordinates & id ) to call at each iteration.
 *                                Where id represents the absolute coordinates of the item to process.
 * @param[in,out] iterators       Tensor iterators which will be updated by this function before calling lambda_function.
 */
template <typename L, typename... Ts>
inline void execute_window_loop(const Window &w, L &&lambda_function, Ts &&... iterators);

/** Update window and padding size for each of the access patterns.
 *
 * First the window size is reduced based on all access patterns that are not
 * allowed to modify the padding of the underlying tensor. Then the padding of
 * the remaining tensors is increased to match the window.
 *
 * @param[in] win      Window that is used by the kernel.
 * @param[in] patterns Access patterns used to calculate the final window and padding.
 *
 * @return True if the window has been changed. Changes to the padding do not
 *         influence the returned value.
 */
template <typename... Ts>
bool update_window_and_padding(Window &win, Ts &&... patterns)
{
    bool window_changed = false;

    for_each([&](const IAccessWindow & w)
    {
        window_changed |= w.update_window_if_needed(win);
    },
    patterns...);

    bool padding_changed = false;

    for_each([&](const IAccessWindow & w)
    {
        padding_changed |= w.update_padding_if_needed(win);
    },
    patterns...);

    return window_changed;
}

/** Calculate the maximum window for a given tensor shape and border setting
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] skip_border (Optional) If true exclude the border region from the window.
 * @param[in] border_size (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_window(const ITensorInfo &info, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize());

/** Calculate the maximum window used by a horizontal kernel for a given tensor shape and border setting
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] skip_border (Optional) If true exclude the border region from the window.
 * @param[in] border_size (Optional) Border size. The border region will be excluded from the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_window_horizontal(const ITensorInfo &info, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize());

/** Calculate the maximum window for a given tensor shape and border setting. The window will also includes the border.
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] border_size (Optional) Border size. The border region will be included in the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_enlarged_window(const ITensorInfo &info, const Steps &steps = Steps(), BorderSize border_size = BorderSize());

/** Intersect multiple valid regions.
 *
 * @param[in] regions Valid regions.
 *
 * @return Intersection of all regions.
 */
template <typename... Ts>
ValidRegion intersect_valid_regions(Ts &&... regions)
{
    auto intersect = [](const ValidRegion & r1, const ValidRegion & r2) -> ValidRegion
    {
        ValidRegion region;

        for(size_t d = 0; d < std::min(r1.anchor.num_dimensions(), r2.anchor.num_dimensions()); ++d)
        {
            region.anchor.set(d, std::max(r1.anchor[d], r2.anchor[d]));
        }

        for(size_t d = 0; d < std::min(r1.shape.num_dimensions(), r2.shape.num_dimensions()); ++d)
        {
            region.shape.set(d, std::min(r1.shape[d], r2.shape[d]));
        }

        return region;
    };

    return foldl(intersect, std::forward<Ts>(regions)...);
}

/** Create a strides object based on the provided strides and the tensor dimensions.
 *
 * @param[in] info          Tensor info object providing the shape of the tensor for unspecified strides.
 * @param[in] stride_x      Stride to be used in X dimension (in bytes).
 * @param[in] fixed_strides Strides to be used in higher dimensions starting at Y (in bytes).
 *
 * @return Strides object based on the specified strides. Missing strides are
 *         calculated based on the tensor shape and the strides of lower dimensions.
 */
template <typename T, typename... Ts>
inline Strides compute_strides(const ITensorInfo &info, T stride_x, Ts &&... fixed_strides)
{
    const TensorShape &shape = info.tensor_shape();

    // Create strides object
    Strides strides(stride_x, fixed_strides...);

    for(size_t i = 1 + sizeof...(Ts); i < info.num_dimensions(); ++i)
    {
        strides.set(i, shape[i - 1] * strides[i - 1]);
    }

    return strides;
}

/** Create a strides object based on the tensor dimensions.
 *
 * @param[in] info Tensor info object used to compute the strides.
 *
 * @return Strides object based on element size and tensor shape.
 */
template <typename... Ts>
inline Strides compute_strides(const ITensorInfo &info)
{
    return compute_strides(info, info.element_size());
}

/** Permutes given Dimensions according to a permutation vector
 *
 * @warning Validity of permutation is not checked
 *
 * @param[in, out] dimensions Dimensions to permute
 * @param[in]      perm       Permutation vector
 */
template <typename T>
inline void permute(Dimensions<T> &dimensions, const PermutationVector &perm)
{
    auto copy_dimensions = utility::make_array<Dimensions<T>::num_max_dimensions>(dimensions.begin(), dimensions.end());
    for(unsigned int i = 0; i < perm.num_dimensions(); ++i)
    {
        dimensions[i] = copy_dimensions[perm[i]];
    }
}

/* Auto initialize the tensor info (shape, number of channels, data type and fixed point position) if the current assignment is empty.
 *
 * @param[in,out] info                 Tensor info used to check and assign.
 * @param[in]     shape                New shape.
 * @param[in]     num_channels         New number of channels.
 * @param[in]     data_type            New data type
 * @param[in]     fixed_point_position New fixed point position
 * @param[in]     quantization_info    (Optional) New quantization info
 *
 * @return True if the tensor info has been initialized
 */
bool auto_init_if_empty(ITensorInfo       &info,
                        const TensorShape &shape,
                        int num_channels, DataType data_type,
                        int              fixed_point_position,
                        QuantizationInfo quantization_info = QuantizationInfo());

/** Auto initialize the tensor info using another tensor info.
 *
 * @param info_sink   Tensor info used to check and assign
 * @param info_source Tensor info used to assign
 *
 * @return True if the tensor info has been initialized
 */
bool auto_init_if_empty(ITensorInfo &info_sink, const ITensorInfo &info_source);

/* Set the shape to the specified value if the current assignment is empty.
 *
 * @param[in,out] info  Tensor info used to check and assign.
 * @param[in]     shape New shape.
 *
 * @return True if the shape has been changed.
 */
bool set_shape_if_empty(ITensorInfo &info, const TensorShape &shape);

/* Set the format, data type and number of channels to the specified value if
 * the current data type is unknown.
 *
 * @param[in,out] info   Tensor info used to check and assign.
 * @param[in]     format New format.
 *
 * @return True if the format has been changed.
 */
bool set_format_if_unknown(ITensorInfo &info, Format format);

/* Set the data type and number of channels to the specified value if
 * the current data type is unknown.
 *
 * @param[in,out] info      Tensor info used to check and assign.
 * @param[in]     data_type New data type.
 *
 * @return True if the data type has been changed.
 */
bool set_data_type_if_unknown(ITensorInfo &info, DataType data_type);

/* Set the fixed point position to the specified value if
 * the current fixed point position is 0 and the data type is QS8 or QS16
 *
 * @param[in,out] info                 Tensor info used to check and assign.
 * @param[in]     fixed_point_position New fixed point position
 *
 * @return True if the fixed point position has been changed.
 */
bool set_fixed_point_position_if_zero(ITensorInfo &info, int fixed_point_position);

/* Set the quantization info to the specified value if
 * the current quantization info is empty and the data type of asymmetric quantized type
 *
 * @param[in,out] info              Tensor info used to check and assign.
 * @param[in]     quantization_info Quantization info
 *
 * @return True if the quantization info has been changed.
 */
bool set_quantization_info_if_empty(ITensorInfo &info, QuantizationInfo quantization_info);

/** Helper function to calculate the Valid Region for Scale.
 *
 * @param[in] src_info         Input tensor info used to check.
 * @param[in] dst_shape        Shape of the output.
 * @param[in] policy           Interpolation policy.
 * @param[in] border_size      Size of the border.
 * @param[in] border_undefined True if the border is undefined.
 *
 * @return The corrispondent valid region
 */
ValidRegion calculate_valid_region_scale(const ITensorInfo &src_info, const TensorShape &dst_shape, InterpolationPolicy policy, BorderSize border_size, bool border_undefined);

/** Convert a linear index into n-dimensional coordinates.
 *
 * @param[in] shape Shape of the n-dimensional tensor.
 * @param[in] index Linear index specifying the i-th element.
 *
 * @return n-dimensional coordinates.
 */
inline Coordinates index2coords(const TensorShape &shape, int index);

/** Convert n-dimensional coordinates into a linear index.
 *
 * @param[in] shape Shape of the n-dimensional tensor.
 * @param[in] coord N-dimensional coordinates.
 *
 * @return linead index
 */
inline int coords2index(const TensorShape &shape, const Coordinates &coord);
} // namespace arm_compute

#include "arm_compute/core/Helpers.inl"
#endif /*__ARM_COMPUTE_HELPERS_H__ */
