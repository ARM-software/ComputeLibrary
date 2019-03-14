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
#ifndef __ARM_COMPUTE_HELPERS_H__
#define __ARM_COMPUTE_HELPERS_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Steps.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

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

/** Disable bitwise operations by default */
template <typename T>
struct enable_bitwise_ops
{
    static constexpr bool value = false; /**< Disabled */
};

#ifndef DOXYGEN_SKIP_THIS
template <typename T>
typename std::enable_if<enable_bitwise_ops<T>::value, T>::type operator&(T lhs, T rhs)
{
    using underlying_type = typename std::underlying_type<T>::type;
    return static_cast<T>(static_cast<underlying_type>(lhs) & static_cast<underlying_type>(rhs));
}
#endif /* DOXYGEN_SKIP_THIS */

/** Helper function to create and return a unique_ptr pointed to a CL/GLES kernel object
 *  It also calls the kernel's configuration.
 *
 * @param[in] args All the arguments that need pass to kernel's configuration.
 *
 * @return A unique pointer pointed to a CL/GLES kernel object
 */
template <typename Kernel, typename... T>
std::unique_ptr<Kernel> create_configure_kernel(T &&... args)
{
    std::unique_ptr<Kernel> k = arm_compute::support::cpp14::make_unique<Kernel>();
    k->configure(std::forward<T>(args)...);
    return k;
}

/** Helper function to create and return a unique_ptr pointed to a CL/GLES kernel object
 *
 * @return A unique pointer pointed to a Kernel kernel object
 */
template <typename Kernel>
std::unique_ptr<Kernel> create_kernel()
{
    std::unique_ptr<Kernel> k = arm_compute::support::cpp14::make_unique<Kernel>();
    return k;
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

/** Computes bilinear interpolation for quantized input and output, using the pointer to the top-left pixel and the pixel's distance between
 * the real coordinates and the smallest following integer coordinates. Input must be quantized and in single channel format.
 *
 * @param[in] pixel_ptr Pointer to the top-left pixel value of a single channel input.
 * @param[in] stride    Stride to access the bottom-left and bottom-right pixel values
 * @param[in] dx        Pixel's distance between the X real coordinate and the smallest X following integer
 * @param[in] dy        Pixel's distance between the Y real coordinate and the smallest Y following integer
 * @param[in] iq_info   Input QuantizationInfo
 * @param[in] oq_info   Output QuantizationInfo
 *
 * @note dx and dy must be in the range [0, 1.0]
 *
 * @return The bilinear interpolated pixel value
 */
inline uint8_t delta_bilinear_c1_quantized(const uint8_t *pixel_ptr, size_t stride, float dx, float dy, QuantizationInfo iq_info, QuantizationInfo oq_info)
{
    ARM_COMPUTE_ERROR_ON(pixel_ptr == nullptr);

    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;

    const float a00 = iq_info.dequantize(*pixel_ptr);
    const float a01 = iq_info.dequantize(*(pixel_ptr + 1));
    const float a10 = iq_info.dequantize(*(pixel_ptr + stride));
    const float a11 = iq_info.dequantize(*(pixel_ptr + stride + 1));

    const float w1  = dx1 * dy1;
    const float w2  = dx * dy1;
    const float w3  = dx1 * dy;
    const float w4  = dx * dy;
    float       res = a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4;
    return static_cast<uint8_t>(oq_info.quantize(res, RoundingPolicy::TO_NEAREST_UP));
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

    utility::for_each([&](const IAccessWindow & w)
    {
        window_changed |= w.update_window_if_needed(win);
    },
    patterns...);

    bool padding_changed = false;

    utility::for_each([&](IAccessWindow & w)
    {
        padding_changed |= w.update_padding_if_needed(win);
    },
    patterns...);

    return window_changed;
}

/** Calculate the maximum window for a given tensor shape and border setting
 *
 * @param[in] valid_region Valid region object defining the shape of the tensor space for which the window is created.
 * @param[in] steps        (Optional) Number of elements processed for each step.
 * @param[in] skip_border  (Optional) If true exclude the border region from the window.
 * @param[in] border_size  (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_window(const ValidRegion &valid_region, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize());

/** Calculate the maximum window for a given tensor shape and border setting
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] skip_border (Optional) If true exclude the border region from the window.
 * @param[in] border_size (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
inline Window calculate_max_window(const ITensorInfo &info, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize())
{
    return calculate_max_window(info.valid_region(), steps, skip_border, border_size);
}

/** Calculate the maximum window used by a horizontal kernel for a given tensor shape and border setting
 *
 * @param[in] valid_region Valid region object defining the shape of the tensor space for which the window is created.
 * @param[in] steps        (Optional) Number of elements processed for each step.
 * @param[in] skip_border  (Optional) If true exclude the border region from the window.
 * @param[in] border_size  (Optional) Border size. The border region will be excluded from the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_window_horizontal(const ValidRegion &valid_region, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize());

/** Calculate the maximum window used by a horizontal kernel for a given tensor shape and border setting
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] skip_border (Optional) If true exclude the border region from the window.
 * @param[in] border_size (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
inline Window calculate_max_window_horizontal(const ITensorInfo &info, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize())
{
    return calculate_max_window_horizontal(info.valid_region(), steps, skip_border, border_size);
}

/** Calculate the maximum window for a given tensor shape and border setting. The window will also includes the border.
 *
 * @param[in] valid_region Valid region object defining the shape of the tensor space for which the window is created.
 * @param[in] steps        (Optional) Number of elements processed for each step.
 * @param[in] border_size  (Optional) Border size. The border region will be included in the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_enlarged_window(const ValidRegion &valid_region, const Steps &steps = Steps(), BorderSize border_size = BorderSize());

/** Calculate the maximum window for a given tensor shape and border setting. The window will also includes the border.
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] border_size (Optional) Border size. The border region will be included in the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
inline Window calculate_max_enlarged_window(const ITensorInfo &info, const Steps &steps = Steps(), BorderSize border_size = BorderSize())
{
    return calculate_max_enlarged_window(info.valid_region(), steps, border_size);
}

/** Intersect multiple valid regions.
 *
 * @param[in] regions Valid regions.
 *
 * @return Intersection of all regions.
 */
template <typename... Ts>
ValidRegion intersect_valid_regions(const Ts &... regions)
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

    return utility::foldl(intersect, regions...);
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
    auto dimensions_copy = utility::make_array<Dimensions<T>::num_max_dimensions>(dimensions.begin(), dimensions.end());
    for(unsigned int i = 0; i < perm.num_dimensions(); ++i)
    {
        T dimension_val = (perm[i] < dimensions.num_dimensions()) ? dimensions_copy[perm[i]] : 0;
        dimensions.set(i, dimension_val);
    }
}

/** Permutes given TensorShape according to a permutation vector
 *
 * @warning Validity of permutation is not checked
 *
 * @param[in, out] shape Shape to permute
 * @param[in]      perm  Permutation vector
 */
inline void permute(TensorShape &shape, const PermutationVector &perm)
{
    TensorShape shape_copy = shape;
    for(unsigned int i = 0; i < perm.num_dimensions(); ++i)
    {
        size_t dimension_val = (perm[i] < shape.num_dimensions()) ? shape_copy[perm[i]] : 1;
        shape.set(i, dimension_val, false); // Avoid changes in _num_dimension
    }
}

/** Auto initialize the tensor info (shape, number of channels and data type) if the current assignment is empty.
 *
 * @param[in,out] info              Tensor info used to check and assign.
 * @param[in]     shape             New shape.
 * @param[in]     num_channels      New number of channels.
 * @param[in]     data_type         New data type
 * @param[in]     quantization_info (Optional) New quantization info
 *
 * @return True if the tensor info has been initialized
 */
bool auto_init_if_empty(ITensorInfo       &info,
                        const TensorShape &shape,
                        int num_channels, DataType data_type,
                        QuantizationInfo quantization_info = QuantizationInfo());

/** Auto initialize the tensor info using another tensor info.
 *
 * @param info_sink   Tensor info used to check and assign
 * @param info_source Tensor info used to assign
 *
 * @return True if the tensor info has been initialized
 */
bool auto_init_if_empty(ITensorInfo &info_sink, const ITensorInfo &info_source);

/** Set the shape to the specified value if the current assignment is empty.
 *
 * @param[in,out] info  Tensor info used to check and assign.
 * @param[in]     shape New shape.
 *
 * @return True if the shape has been changed.
 */
bool set_shape_if_empty(ITensorInfo &info, const TensorShape &shape);

/** Set the format, data type and number of channels to the specified value if
 * the current data type is unknown.
 *
 * @param[in,out] info   Tensor info used to check and assign.
 * @param[in]     format New format.
 *
 * @return True if the format has been changed.
 */
bool set_format_if_unknown(ITensorInfo &info, Format format);

/** Set the data type and number of channels to the specified value if
 * the current data type is unknown.
 *
 * @param[in,out] info      Tensor info used to check and assign.
 * @param[in]     data_type New data type.
 *
 * @return True if the data type has been changed.
 */
bool set_data_type_if_unknown(ITensorInfo &info, DataType data_type);

/** Set the data layout to the specified value if
 * the current data layout is unknown.
 *
 * @param[in,out] info        Tensor info used to check and assign.
 * @param[in]     data_layout New data layout.
 *
 * @return True if the data type has been changed.
 */
bool set_data_layout_if_unknown(ITensorInfo &info, DataLayout data_layout);

/** Set the quantization info to the specified value if
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
 * @param[in] src_info           Input tensor info used to check.
 * @param[in] dst_shape          Shape of the output.
 * @param[in] interpolate_policy Interpolation policy.
 * @param[in] sampling_policy    Sampling policy.
 * @param[in] border_undefined   True if the border is undefined.
 *
 * @return The corresponding valid region
 */
ValidRegion calculate_valid_region_scale(const ITensorInfo &src_info, const TensorShape &dst_shape,
                                         InterpolationPolicy interpolate_policy, SamplingPolicy sampling_policy, bool border_undefined);

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

/** Get the index of the given dimension.
 *
 * @param[in] data_layout           The data layout.
 * @param[in] data_layout_dimension The dimension which this index is requested for.
 *
 * @return The int conversion of the requested data layout index.
 */
inline size_t get_data_layout_dimension_index(const DataLayout data_layout, const DataLayoutDimension data_layout_dimension);

/** Get the DataLayoutDimension of a given index and layout.
 *
 * @param[in] data_layout The data layout.
 * @param[in] index       The data layout index.
 *
 * @return The dimension which this index is requested for.
 */
inline DataLayoutDimension get_index_data_layout_dimension(const DataLayout data_layout, const size_t index);

/** Calculate the normalization dimension index for a given normalization type
 *
 * @param[in] layout Data layout of the input and output tensor
 * @param[in] info   Normalization info
 *
 * @return Normalization dimension index
 */
inline unsigned int get_normalization_dimension_index(DataLayout layout, const NormalizationLayerInfo &info)
{
    const unsigned int width_idx   = get_data_layout_dimension_index(layout, DataLayoutDimension::WIDTH);
    const unsigned int channel_idx = get_data_layout_dimension_index(layout, DataLayoutDimension::CHANNEL);

    return info.is_in_map() ? width_idx : channel_idx;
}

/** Calculate the number of output tiles required by Winograd Convolution layer. This utility function can be used by the Winograd input transform
 *  to know the number of tiles on the x and y direction
 *
 * @param[in] in_dims          Spatial dimensions of the input tensor of convolution layer
 * @param[in] kernel_size      Kernel size
 * @param[in] output_tile_size Size of a single output tile
 * @param[in] conv_info        Convolution info (i.e. pad, stride,...)
 *
 * @return the number of output tiles along the x and y directions of size "output_tile_size"
 */
inline Size2D compute_winograd_convolution_tiles(const Size2D &in_dims, const Size2D &kernel_size, const Size2D &output_tile_size, const PadStrideInfo &conv_info)
{
    int num_tiles_x = std::ceil((in_dims.width - (kernel_size.width - 1) + conv_info.pad_left() + conv_info.pad_right()) / static_cast<float>(output_tile_size.width));
    int num_tiles_y = std::ceil((in_dims.height - (kernel_size.height - 1) + conv_info.pad_top() + conv_info.pad_bottom()) / static_cast<float>(output_tile_size.height));

    // Clamp in case we provide paddings but we have 1D convolution
    num_tiles_x = std::min(num_tiles_x, static_cast<int>(in_dims.width));
    num_tiles_y = std::min(num_tiles_y, static_cast<int>(in_dims.height));

    return Size2D(num_tiles_x, num_tiles_y);
}

/** Wrap-around a number within the range 0 <= x < m
 *
 * @param[in] x Input value
 * @param[in] m Range
 *
 * @return the wrapped-around number
 */
template <typename T>
inline T wrap_around(T x, T m)
{
    return x >= 0 ? x % m : (x % m + m) % m;
}

/** Given an integer value, this function returns the next power of two
 *
 * @param[in] x Input value
 *
 * @return the next power of two
 */
inline unsigned int get_next_power_two(unsigned int x)
{
    // Decrement by 1
    x--;

    // Shift right by 1
    x |= x >> 1u;
    // Shift right by 2
    x |= x >> 2u;
    // Shift right by 4
    x |= x >> 4u;
    // Shift right by 8
    x |= x >> 8u;
    // Shift right by 16
    x |= x >> 16u;

    // Increment by 1
    x++;

    return x;
}
} // namespace arm_compute

#include "arm_compute/core/Helpers.inl"
#endif /*__ARM_COMPUTE_HELPERS_H__ */
