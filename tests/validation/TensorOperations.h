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
#ifndef __ARM_COMPUTE_TEST_TENSOR_OPERATIONS_H__
#define __ARM_COMPUTE_TEST_TENSOR_OPERATIONS_H__

#include "FixedPoint.h"
#include "Tensor.h"
#include "Types.h"
#include "Utils.h"
#include "support/ToolchainSupport.h"

#include "FixedPoint.h"
#include "Types.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"
#include "tests/validation/ValidationUserConfiguration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#if ARM_COMPUTE_ENABLE_FP16
//Beware! most std templates acting on types don't work with the data type float16_t
namespace std
{
template <>
class numeric_limits<float16_t>
{
public:
    static float16_t lowest()
    {
        return -std::numeric_limits<float>::max(); // -inf
    };
    static float16_t max()
    {
        return std::numeric_limits<float>::max(); // +inf
    };
};
}
#endif /* ARM_COMPUTE_ENABLE_FP16 */

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace tensor_operations
{
namespace
{
template <class T>
struct is_floating_point
    : std::integral_constant < bool,
      std::is_same<float, typename std::remove_cv<T>::type>::value ||
#ifdef ARM_COMPUTE_ENABLE_FP16
      std::is_same<float16_t, typename std::remove_cv<T>::type>::value ||
#endif /* ARM_COMPUTE_ENABLE_FP16 */
      std::is_same<double, typename std::remove_cv<T>::type>::value || std::is_same<long double, typename std::remove_cv<T>::type>::value >
{
};

bool is_valid_pixel(int i, int min, int max)
{
    return (i >= min && i < max);
}

// 3D convolution for floating point type
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void convolution3d(const T *in, const T *weights, const T *bias, T *out, int xi, int yi, int width_in, int height_in, int depth_in, int width_weights, int height_weights, int8_t fixed_point_position)
{
    const int half_width_weights  = width_weights / 2;
    const int half_height_weights = height_weights / 2;

    // Reset accumulator
    T acc = static_cast<T>(0);

    // Compute a 2D convolution for each IFM and accumulate the result
    for(int ifm = 0; ifm < depth_in; ++ifm)
    {
        // Compute the offset for the input slice
        const int offset_slice_in = xi + yi * width_in + ifm * width_in * height_in;

        // Compute 2D convolution
        for(int yk = -half_height_weights; yk <= half_height_weights; ++yk)
        {
            for(int xk = -half_width_weights; xk <= half_width_weights; ++xk)
            {
                // Check if the pixel is out-of-bound
                if(is_valid_pixel(xi + xk, 0, width_in) && is_valid_pixel(yi + yk, 0, height_in))
                {
                    const int idx = xk + half_width_weights;
                    const int idy = yk + half_height_weights;

                    const T i_value = in[offset_slice_in + xk + yk * width_in];
                    const T w_value = weights[idx + idy * width_weights + ifm * width_weights * height_weights];

                    acc += i_value * w_value;
                }
            }
        }
    }

    // Accumulate the bias and store the result
    *out = acc + (*bias);
}

// 3D convolution for fixed point type
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void convolution3d(const T *in, const T *weights, const T *bias, T *out, int xi, int yi, int width_in, int height_in, int depth_in, int width_weights, int height_weights,
                   int8_t fixed_point_position)
{
    const int half_width_weights  = width_weights / 2;
    const int half_height_weights = height_weights / 2;

    using namespace fixed_point_arithmetic;
    using promoted_type = typename fixed_point_arithmetic::traits::promote<T>::type;

    // Reset accumulator
    fixed_point<promoted_type> acc(0, fixed_point_position);

    // Compute a 2D convolution for each IFM and accumulate the result
    for(int ifm = 0; ifm < depth_in; ++ifm)
    {
        // Compute the offset for the input slice
        const int offset_slice_in = xi + yi * width_in + ifm * width_in * height_in;

        // Compute 2D convolution
        for(int yk = -half_height_weights; yk <= half_height_weights; ++yk)
        {
            for(int xk = -half_width_weights; xk <= half_width_weights; ++xk)
            {
                // Check if the pixel is out-of-bound
                if(is_valid_pixel(xi + xk, 0, width_in) && is_valid_pixel(yi + yk, 0, height_in))
                {
                    const int idx = xk + half_width_weights;
                    const int idy = yk + half_height_weights;

                    const fixed_point<promoted_type> i_value(in[offset_slice_in + xk + yk * width_in], fixed_point_position, true);
                    const fixed_point<promoted_type> w_value(weights[idx + idy * width_weights + ifm * width_weights * height_weights], fixed_point_position, true);
                    const fixed_point<promoted_type> iw = i_value * w_value;
                    acc                                 = iw + acc;
                }
            }
        }
    }

    // Get the bias
    const fixed_point<promoted_type> b(*bias, fixed_point_position, true);

    // Accumulate the bias and covert back
    acc = acc + b;
    fixed_point<T> res(acc);
    *out = res.raw();
}

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void vector_matrix_multiply(const T *in, const T *weights, const T *bias, T *out, int cols_weights, int rows_weights, uint8_t fixed_point_position)
{
    for(int x = 0; x < cols_weights; ++x)
    {
        T acc = 0.0f;
        for(int y = 0; y < rows_weights; ++y)
        {
            acc += in[y] * weights[x + y * cols_weights];
        }
        out[x] = acc + bias[x];
    }
}

// Vector matrix multiply for fixed point type
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void vector_matrix_multiply(const T *in, const T *weights, const T *bias, T *out, int cols_weights, int rows_weights, uint8_t fixed_point_position)
{
    using namespace fixed_point_arithmetic;
    using promoted_type = typename fixed_point_arithmetic::traits::promote<T>::type;

    for(int x = 0; x < cols_weights; ++x)
    {
        // Reset accumulator
        fixed_point<promoted_type> acc(0, fixed_point_position);

        for(int y = 0; y < rows_weights; ++y)
        {
            const fixed_point<promoted_type> i_value(in[y], fixed_point_position, true);
            const fixed_point<promoted_type> w_value(weights[x + y * cols_weights], fixed_point_position, true);
            const fixed_point<promoted_type> iw = i_value * w_value;
            acc                                 = iw + acc;
        }

        // Get the bias
        const fixed_point<T> b(bias[x], fixed_point_position, true);

        // Convert back and accumulate the bias
        fixed_point<T> res(acc);
        res = res + b;

        // Store the result
        out[x] = res.raw();
    }
}

// Return a tensor element at a specified coordinate with different border modes
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
T tensor_elem_at(const Tensor<T> &in, Coordinates &coord, BorderMode border_mode, T constant_border_value)
{
    const int x      = coord.x();
    const int y      = coord.y();
    const int width  = static_cast<int>(in.shape().x());
    const int height = static_cast<int>(in.shape().y());

    // If coordinates beyond range of tensor's width or height
    if(x < 0 || y < 0 || x >= width || y >= height)
    {
        if(border_mode == BorderMode::REPLICATE)
        {
            coord.set(0, std::max(0, std::min(x, width - 1)));
            coord.set(1, std::max(0, std::min(y, height - 1)));
            return in[coord2index(in.shape(), coord)];
        }
        else
        {
            return constant_border_value;
        }
    }
    else
    {
        return in[coord2index(in.shape(), coord)];
    }
}

/** Apply 2D spatial filter on a single element of @p in at coordinates @p coord
 *
 * - filter sizes have to be odd number
 * - Row major order of filter assumed
 * - TO_ZERO rounding policy assumed
 * - SATURATE convert policy assumed
 *
 */
template <typename T1, typename T2, typename T3>
void apply_2d_spatial_filter(Coordinates coord, const Tensor<T1> &in, Tensor<T3> &out, const TensorShape &filter_shape, const T2 *filter_itr, float scale, BorderMode border_mode,
                             T1 constant_border_value = 0)
{
    double    val = 0;
    const int x   = coord.x();
    const int y   = coord.y();
    for(int j = y - static_cast<int>(filter_shape[1] / 2); j <= y + static_cast<int>(filter_shape[1] / 2); ++j)
    {
        for(int i = x - static_cast<int>(filter_shape[0] / 2); i <= x + static_cast<int>(filter_shape[0] / 2); ++i)
        {
            coord.set(0, i);
            coord.set(1, j);
            val += static_cast<double>(*filter_itr) * tensor_elem_at(in, coord, border_mode, constant_border_value);
            ++filter_itr;
        }
    }
    coord.set(0, x);
    coord.set(1, y);
    const double rounded_val = support::cpp11::trunc(val * static_cast<double>(scale));
    out[coord2index(in.shape(), coord)] = saturate_cast<T3>(rounded_val);
}
} // namespace

// Sobel 3x3
template <typename T1, typename T2>
void sobel_3x3(Tensor<T1> &in, Tensor<T2> &out_x, Tensor<T2> &out_y, BorderMode border_mode, uint8_t constant_border_value)
{
    const std::array<int8_t, 9> sobel_x{ { -1, 0, 1, -2, 0, 2, -1, 0, 1 } };
    const std::array<int8_t, 9> sobel_y{ { -1, -2, -1, 0, 0, 0, 1, 2, 1 } };

    for(int element_idx = 0; element_idx < in.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(in.shape(), element_idx);

        apply_2d_spatial_filter(id, in, out_x, TensorShape(3U, 3U), sobel_x.data(), 1.f, border_mode, constant_border_value);
        apply_2d_spatial_filter(id, in, out_y, TensorShape(3U, 3U), sobel_y.data(), 1.f, border_mode, constant_border_value);
    }
}

// Sobel 5x5
template <typename T1, typename T2>
void sobel_5x5(Tensor<T1> &in, Tensor<T2> &out_x, Tensor<T2> &out_y, BorderMode border_mode, uint8_t constant_border_value)
{
    const std::array<int8_t, 25> sobel_x{ {
            -1, -2, 0, 2, 1,
            -4, -8, 0, 8, 4,
            -6, -12, 0, 12, 6,
            -4, -8, 0, 8, 4,
            -1, -2, 0, 2, 1
        } };

    const std::array<int8_t, 25> sobel_y{ {
            -1, -4, -6, -4, -1,
            -2, -8, -12, -8, -2,
            0, 0, 0, 0, 0,
            2, 8, 12, 8, 2,
            1, 4, 6, 4, 1
        } };

    for(int element_idx = 0; element_idx < in.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(in.shape(), element_idx);

        apply_2d_spatial_filter(id, in, out_x, TensorShape(5U, 5U), sobel_x.data(), 1.f, border_mode, constant_border_value);
        apply_2d_spatial_filter(id, in, out_y, TensorShape(5U, 5U), sobel_y.data(), 1.f, border_mode, constant_border_value);
    }
}

// Min max location
template <typename T1>
void min_max_location(const Tensor<T1> &in, int32_t &min, int32_t &max, IArray<Coordinates2D> &min_loc, IArray<Coordinates2D> &max_loc, uint32_t &min_count, uint32_t &max_count)
{
    // Set min and max to first pixel
    min       = in[0];
    max       = in[0];
    min_count = 0;
    max_count = 0;

    const size_t width = in.shape().x();

    // Look for min and max values
    for(int i = 1; i < in.num_elements(); ++i)
    {
        if(static_cast<int32_t>(in[i]) < min)
        {
            min = in[i];
        }
        if(static_cast<int32_t>(in[i]) > max)
        {
            max = in[i];
        }
    }

    for(int i = 0; i < in.num_elements(); ++i)
    {
        if(static_cast<int32_t>(in[i]) == min)
        {
            Coordinates2D min_coord;
            min_coord.x = static_cast<int32_t>(i % width);
            min_coord.y = static_cast<int32_t>(i / width);

            min_loc.push_back(min_coord);

            min_count++;
        }
        if(static_cast<int32_t>(in[i]) == max)
        {
            Coordinates2D max_coord;
            max_coord.x = static_cast<int32_t>(i % width);
            max_coord.y = static_cast<int32_t>(i / width);

            max_loc.push_back(max_coord);

            max_count++;
        }
    }
}

// Mean Standard Deviation
template <typename T1>
void mean_and_standard_deviation(const Tensor<T1> &in, float &mean, float &std_dev)
{
    int num_elements = in.num_elements();

    // Calculate mean
    mean = 0.f;
    for(int i = 0; i < num_elements; ++i)
    {
        mean += in[i];
    }
    mean /= num_elements;

    // Calculate standard deviation
    std_dev = 0.f;
    for(int i = 0; i < num_elements; ++i)
    {
        std_dev += (mean - in[i]) * (mean - in[i]);
    }
    std_dev = sqrt(std_dev / num_elements);
}

// Integral Image
void integral_image(const Tensor<uint8_t> &in, Tensor<uint32_t> &out)
{
    // Length of dimensions
    const size_t width  = in.shape().x();
    const size_t height = in.shape().y();
    const size_t depth  = in.shape().z() * in.shape()[3] * in.shape()[4] * in.shape()[5];

    const size_t image_size = width * height;

    for(size_t z = 0; z < depth; ++z)
    {
        size_t current_image = z * image_size;

        //First element of each image
        out[current_image] = in[current_image];

        // First row of each image (add only pixel on the left)
        for(size_t x = 1; x < width; ++x)
        {
            out[current_image + x] = static_cast<uint32_t>(in[current_image + x]) + out[current_image + x - 1];
        }

        // Subsequent rows
        for(size_t y = 1; y < height; ++y)
        {
            size_t current_row = current_image + (width * y);

            // First element of each row (add only pixel up)
            out[current_row] = static_cast<uint32_t>(in[current_row]) + out[current_row - width];

            // Following row elements
            for(size_t x = 1; x < width; ++x)
            {
                size_t current_pixel = current_row + x;

                // out = in + up(out) + left(out) - up_left(out)
                out[current_pixel] = static_cast<uint32_t>(in[current_pixel]) + out[current_pixel - 1]
                                     + out[current_pixel - width] - out[current_pixel - width - 1];
            }
        }
    }
}

// Absolute difference
template <typename T1, typename T2, typename T3>
void absolute_difference(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out)
{
    using intermediate_type = typename common_promoted_signed_type<T1, T2, T3>::intermediate_type;

    for(int i = 0; i < in1.num_elements(); ++i)
    {
        intermediate_type val = std::abs(static_cast<intermediate_type>(in1[i]) - static_cast<intermediate_type>(in2[i]));
        out[i]                = saturate_cast<T3>(val);
    }
}

// Accumulate
template <typename T1, typename T2>
void accumulate(const Tensor<T1> &in, Tensor<T2> &out)
{
    using intermediate_type = typename common_promoted_signed_type<T1, T2>::intermediate_type;

    for(int i = 0; i < in.num_elements(); ++i)
    {
        intermediate_type val = static_cast<intermediate_type>(out[i]) + static_cast<intermediate_type>(in[i]);
        out[i]                = saturate_cast<T2>(val);
    }
}

// Accumulate squared
template <typename T1, typename T2>
void accumulate_squared(const Tensor<T1> &in, Tensor<T2> &out, uint32_t shift)
{
    if(shift > 15)
    {
        ARM_COMPUTE_ERROR("Shift in accumulate_squared must be within the range [0, 15]");
    }
    using intermediate_type = typename common_promoted_signed_type<T1, T2>::intermediate_type;
    intermediate_type denom = 1 << shift;

    for(int i = 0; i < in.num_elements(); ++i)
    {
        intermediate_type val = static_cast<intermediate_type>(out[i]) + (static_cast<intermediate_type>(in[i]) * static_cast<intermediate_type>(in[i]) / denom);
        out[i]                = saturate_cast<T2>(val);
    }
}

// Accumulate weighted
template <typename T>
void accumulate_weighted(const Tensor<T> &in, Tensor<T> &out, float alpha)
{
    if(alpha < 0.f || alpha > 1.f)
    {
        ARM_COMPUTE_ERROR("Weight (alpha) specified in accumulate_weighted must be within the range [0, 1]");
    }
    using intermediate_type = typename common_promoted_signed_type<T>::intermediate_type;

    for(int i = 0; i < in.num_elements(); ++i)
    {
        double val = (1. - static_cast<double>(alpha)) * static_cast<intermediate_type>(out[i]) + static_cast<double>(alpha) * static_cast<intermediate_type>(in[i]);
        out[i]     = static_cast<T>(val);
    }
}

// Arithmetic addition
template <typename T1, typename T2, typename T3>
void arithmetic_addition(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out, ConvertPolicy convert_policy)
{
    using intermediate_type = typename common_promoted_signed_type<T1, T2, T3>::intermediate_type;

    for(int i = 0; i < in1.num_elements(); ++i)
    {
        intermediate_type val = static_cast<intermediate_type>(in1[i]) + static_cast<intermediate_type>(in2[i]);
        out[i]                = (convert_policy == ConvertPolicy::SATURATE) ? saturate_cast<T3>(val) : static_cast<T3>(val);
    }
}

// Arithmetic Subtraction
template <typename T1, typename T2, typename T3>
void arithmetic_subtraction(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out, ConvertPolicy convert_policy)
{
    using intermediate_type = typename common_promoted_signed_type<T1, T2, T3>::intermediate_type;

    for(int i = 0; i < in1.num_elements(); ++i)
    {
        intermediate_type val = static_cast<intermediate_type>(in1[i]) - static_cast<intermediate_type>(in2[i]);
        out[i]                = (convert_policy == ConvertPolicy::SATURATE) ? saturate_cast<T3>(val) : static_cast<T3>(val);
    }
}

// Bitwise and
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void bitwise_and(const Tensor<T> &in1, const Tensor<T> &in2, Tensor<T> &out)
{
    for(int i = 0; i < in1.num_elements(); ++i)
    {
        out[i] = in1[i] & in2[i];
    }
}

// Bitwise or
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void bitwise_or(const Tensor<T> &in1, const Tensor<T> &in2, Tensor<T> &out)
{
    for(int i = 0; i < in1.num_elements(); ++i)
    {
        out[i] = in1[i] | in2[i];
    }
}

// Bitwise xor
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void bitwise_xor(const Tensor<T> &in1, const Tensor<T> &in2, Tensor<T> &out)
{
    for(int i = 0; i < in1.num_elements(); ++i)
    {
        out[i] = in1[i] ^ in2[i];
    }
}

// Bitwise not
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void bitwise_not(const Tensor<T> &in, Tensor<T> &out)
{
    for(int i = 0; i < in.num_elements(); ++i)
    {
        out[i] = ~in[i];
    }
}

// Box3x3 filter
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void box3x3(const Tensor<T> &in, Tensor<T> &out, BorderMode border_mode, T constant_border_value)
{
    const std::array<T, 9> filter{ { 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
    float scale = 1.f / static_cast<float>(filter.size());
    for(int element_idx = 0; element_idx < in.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(in.shape(), element_idx);
        apply_2d_spatial_filter(id, in, out, TensorShape(3U, 3U), filter.data(), scale, border_mode, constant_border_value);
    }
}

// Depth conversion
template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&is_floating_point<T2>::value, int >::type = 0 >
void depth_convert(const Tensor<T1> &in, Tensor<T2> &out, ConvertPolicy policy, uint32_t shift)
{
    using namespace fixed_point_arithmetic;

    const int fixed_point_position = in.fixed_point_position();
    for(int i = 0; i < in.num_elements(); ++i)
    {
        out[i] = static_cast<float>(fixed_point<T1>(in[i], fixed_point_position, true));
    }
}

template < typename T1, typename T2, typename std::enable_if < is_floating_point<T1>::value &&std::is_integral<T2>::value, int >::type = 0 >
void depth_convert(const Tensor<T1> &in, Tensor<T2> &out, ConvertPolicy policy, uint32_t shift)
{
    using namespace fixed_point_arithmetic;

    const int fixed_point_position = out.fixed_point_position();
    for(int i = 0; i < in.num_elements(); ++i)
    {
        out[i] = fixed_point<T2>(in[i], fixed_point_position).raw();
    }
}

template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&std::is_integral<T2>::value &&!std::is_same<T1, T2>::value, int >::type = 0 >
void depth_convert(const Tensor<T1> &in, Tensor<T2> &out, ConvertPolicy policy, uint32_t shift)
{
    // Up-casting
    if(std::numeric_limits<T1>::digits <= std::numeric_limits<T2>::digits)
    {
        for(int i = 0; i < in.num_elements(); ++i)
        {
            out[i] = static_cast<T2>(in[i]) << shift;
        }
    }
    // Down-casting
    else
    {
        for(int i = 0; i < in.num_elements(); ++i)
        {
            T1 val = in[i] >> shift;
            out[i] = ((policy == ConvertPolicy::SATURATE) ? saturate_cast<T2>(val) : static_cast<T2>(val));
        }
    }
}

template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&std::is_integral<T2>::value &&std::is_same<T1, T2>::value, int >::type = 0 >
void depth_convert(const Tensor<T1> &in, Tensor<T2> &out, ConvertPolicy policy, uint32_t shift)
{
    using namespace fixed_point_arithmetic;
    bool is_in_place = (&in == &out);

    const int fixed_point_position_in  = in.fixed_point_position();
    const int fixed_point_position_out = (is_in_place) ? static_cast<int>(shift) : out.fixed_point_position();

    if(!is_in_place || (fixed_point_position_in != fixed_point_position_out))
    {
        for(int i = 0; i < in.num_elements(); ++i)
        {
            auto x = fixed_point<T2>(in[i], fixed_point_position_in, true);
            x.rescale(fixed_point_position_out);
            out[i] = x.raw();
        }
    }
}

template < typename T1, typename T2, typename std::enable_if < is_floating_point<T1>::value &&is_floating_point<T2>::value, int >::type = 0 >
void depth_convert(const Tensor<T1> &in, Tensor<T2> &out, ConvertPolicy policy, uint32_t shift)
{
    for(int i = 0; i < in.num_elements(); ++i)
    {
        out[i] = static_cast<T2>(in[i]);
    }
}

// Gaussian3x3 filter
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void gaussian3x3(const Tensor<T> &in, Tensor<T> &out, BorderMode border_mode, T constant_border_value)
{
    const std::array<T, 9> filter{ { 1, 2, 1, 2, 4, 2, 1, 2, 1 } };
    const float scale = 1.f / 16.f;
    for(int element_idx = 0; element_idx < in.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(in.shape(), element_idx);
        apply_2d_spatial_filter(id, in, out, TensorShape(3U, 3U), filter.data(), scale, border_mode, constant_border_value);
    }
}

// Gaussian5x5 filter
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void gaussian5x5(const Tensor<T> &in, Tensor<T> &out, BorderMode border_mode, T constant_border_value)
{
    const std::array<T, 25> filter{ {
            1, 4, 6, 4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1, 4, 6, 4, 1
        } };
    const float scale = 1.f / 256.f;
    for(int element_idx = 0; element_idx < in.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(in.shape(), element_idx);
        apply_2d_spatial_filter(id, in, out, TensorShape(5U, 5U), filter.data(), scale, border_mode, constant_border_value);
    }
}

// Matrix multiplication for floating point type
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void gemm(const Tensor<T> &in1, const Tensor<T> &in2, const Tensor<T> &in3, Tensor<T> &out, float alpha, float beta)
{
    const int M = out.shape().y();
    const int N = out.shape().x();
    const int K = in1.shape().x();

    for(int r = 0; r < M; ++r)
    {
        for(int c = 0; c < N; ++c)
        {
            T acc = 0.0f;

            for(int k = 0; k < K; ++k)
            {
                const T a0 = in1[r * K + k];
                const T b0 = in2[k * N + c];

                acc += a0 * b0;
            }

            // Finalize the result: A * B * alpha + C * beta
            const T c0     = in3[c + r * N];
            out[c + r * N] = alpha * acc + beta * c0;
        }
    }
}

// Matrix multiplication for fixed point type
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void gemm(const Tensor<T> &in1, const Tensor<T> &in2, const Tensor<T> &in3, Tensor<T> &out, float alpha, float beta)
{
    using namespace fixed_point_arithmetic;

    using promoted_type = typename fixed_point_arithmetic::traits::promote<T>::type;

    const int    M                    = out.shape().y();
    const int    N                    = out.shape().x();
    const int    K                    = in1.shape().x();
    const int8_t fixed_point_position = static_cast<int8_t>(in1.fixed_point_position());

    const fixed_point<T> alpha_q(alpha, fixed_point_position);
    const fixed_point<T> beta_q(beta, fixed_point_position);

    for(int r = 0; r < M; ++r)
    {
        for(int c = 0; c < N; ++c)
        {
            fixed_point<promoted_type> acc_q(0, fixed_point_position);

            for(int k = 0; k < K; ++k)
            {
                const fixed_point<promoted_type> a0_q(in1[r * K + k], fixed_point_position, true);
                const fixed_point<promoted_type> b0_q(in2[k * N + c], fixed_point_position, true);
                const fixed_point<promoted_type> axb_q = a0_q * b0_q;

                acc_q = axb_q + acc_q;
            }

            // Finalize the result: A * B * alpha + C * beta
            const fixed_point<T> c0_q(in3[c + r * N], fixed_point_position, true);

            fixed_point<T> res_q(acc_q);
            res_q = alpha_q * res_q;
            res_q = (c0_q * beta_q) + res_q;

            // Store the result
            out[c + r * N] = res_q.raw();
        }
    }
}

// Non linear filter
template <typename T>
void non_linear_filter(const Tensor<T> &in, Tensor<T> &out, NonLinearFilterFunction function, unsigned int mask_size,
                       MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(pattern == MatrixPattern::OTHER && mask == nullptr);

    using intermediate_type = typename common_promoted_signed_type<T>::intermediate_type;

    const int                      sq_mask_size   = mask_size * mask_size;
    const int                      half_mask_size = mask_size / 2;
    std::vector<intermediate_type> vals(sq_mask_size);
    intermediate_type              current_value = 0;

    const ValidRegion valid_region = shape_to_valid_region(in.shape(), border_mode == BorderMode::UNDEFINED, BorderSize(half_mask_size));

    for(int element_idx = 0, count = 0, index = 0; element_idx < in.num_elements(); ++element_idx, count = 0, index = 0)
    {
        Coordinates id = index2coord(in.shape(), element_idx);
        if(is_in_valid_region(valid_region, id))
        {
            int idx = id.x();
            int idy = id.y();
            for(int y = idy - half_mask_size; y <= idy + half_mask_size; ++y)
            {
                for(int x = idx - half_mask_size; x <= idx + half_mask_size; ++x, ++index)
                {
                    id.set(0, x);
                    id.set(1, y);
                    current_value = tensor_elem_at(in, id, border_mode, constant_border_value);

                    if(mask[index] == 255)
                    {
                        vals[count] = static_cast<intermediate_type>(current_value);
                        ++count;
                    }
                }
            }
            std::sort(vals.begin(), vals.begin() + count);
            switch(function)
            {
                case NonLinearFilterFunction::MIN:
                    out[element_idx] = saturate_cast<T>(vals[0]);
                    break;
                case NonLinearFilterFunction::MAX:
                    out[element_idx] = saturate_cast<T>(vals[count - 1]);
                    break;
                case NonLinearFilterFunction::MEDIAN:
                    out[element_idx] = saturate_cast<T>(vals[count / 2]);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported NonLinearFilter function.");
            }
        }
    }
}

// Pixel-wise multiplication
template <typename T1, typename T2, typename T3>
void pixel_wise_multiplication(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    if(scale < 0)
    {
        ARM_COMPUTE_ERROR("Scale of pixel-wise multiplication must be non-negative");
    }
    using intermediate_type = typename common_promoted_signed_type<T1, T2, T3>::intermediate_type;
    for(int i = 0; i < in1.num_elements(); ++i)
    {
        double val = static_cast<intermediate_type>(in1[i]) * static_cast<intermediate_type>(in2[i]) * static_cast<double>(scale);
        if(is_floating_point<T3>::value)
        {
            out[i] = val;
        }
        else
        {
            double rounded_val = 0;
            switch(rounding_policy)
            {
                case(RoundingPolicy::TO_ZERO):
                    rounded_val = support::cpp11::trunc(val);
                    break;
                case(RoundingPolicy::TO_NEAREST_UP):
                    rounded_val = round_half_up(val);
                    break;
                case(RoundingPolicy::TO_NEAREST_EVEN):
                    rounded_val = round_half_even(val);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported rounding policy");
            }
            out[i] = (convert_policy == ConvertPolicy::SATURATE) ? saturate_cast<T3>(rounded_val) : static_cast<T3>(rounded_val);
        }
    }
}

// Fixed-point Pixel-wise Multiplication
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void fixed_point_pixel_wise_multiplication(const Tensor<T> &in1, const Tensor<T> &in2, Tensor<T> &out, int scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    using namespace fixed_point_arithmetic;

    const int fixed_point_position = in1.fixed_point_position();

    ARM_COMPUTE_ERROR_ON_MSG(in1.data_type() != in2.data_type() || in1.data_type() != out.data_type(),
                             "Tensors must all have the same DataType");
    ARM_COMPUTE_ERROR_ON_MSG(fixed_point_position != in2.fixed_point_position() || fixed_point_position != out.fixed_point_position(),
                             "Fixed-point position must be the same for both inputs and outputs");

    // Validate fixed_point_position
    ARM_COMPUTE_ERROR_ON((in1.data_type() == DataType::QS8) && (fixed_point_position == 0 || fixed_point_position > 7));
    ARM_COMPUTE_ERROR_ON((in1.data_type() == DataType::QS16) && (fixed_point_position == 0 || fixed_point_position > 15));

    fixed_point<T> fp_scale(scale, fixed_point_position);
    const bool     is_sat     = convert_policy == ConvertPolicy::SATURATE;
    const bool     do_scaling = scale != 1;

    for(int i = 0; i < in1.num_elements(); ++i)
    {
        fixed_point<T> val1(in1[i], fixed_point_position, true);
        fixed_point<T> val2(in2[i], fixed_point_position, true);
        fixed_point<T> res = (is_sat) ? val1 * val2 : mul<OverflowPolicy::WRAP>(val1, val2);
        if(do_scaling)
        {
            res = (is_sat) ? res * fp_scale : mul<OverflowPolicy::WRAP>(res, fp_scale);
        }
        out[i] = res.raw();
    }
}

//Table Lookup
template <typename T, typename T1>
void table_lookup(const Tensor<T> &in, Tensor<T> &out, std::map<T1, T1> &lut)
{
    for(int i = 0; i < in.num_elements(); ++i)
    {
        out[i] = static_cast<T>(lut[in[i]]);
    }
}

// Threshold
template <typename T>
void threshold(const Tensor<T> &in, Tensor<T> &out, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    switch(type)
    {
        case ThresholdType::BINARY:
            for(int i = 0; i < in.num_elements(); ++i)
            {
                out[i] = ((in[i] > threshold) ? true_value : false_value);
            }
            break;
        case ThresholdType::RANGE:
            for(int i = 0; i < in.num_elements(); ++i)
            {
                if(in[i] > upper)
                {
                    out[i] = false_value;
                }
                else if(in[i] < threshold)
                {
                    out[i] = false_value;
                }
                else
                {
                    out[i] = true_value;
                }
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Thresholding type not recognised");
            break;
    }
}

// Activation Layer for floating point type
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void activation_layer(const Tensor<T> &in, Tensor<T> &out, ActivationLayerInfo act_info)
{
    const T a = static_cast<T>(act_info.a());
    const T b = static_cast<T>(act_info.b());

    for(int i = 0; i < in.num_elements(); ++i)
    {
        T x = in[i];
        switch(act_info.activation())
        {
            case ActivationLayerInfo::ActivationFunction::ABS:
                out[i] = std::abs(x);
                break;
            case ActivationLayerInfo::ActivationFunction::LINEAR:
                out[i] = a * x + b;
                break;
            case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                out[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
                break;
            case ActivationLayerInfo::ActivationFunction::RELU:
                out[i] = std::max<T>(0, x);
                break;
            case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                out[i] = std::min<T>(a, std::max<T>(0, x));
                break;
            case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                out[i] = (x > 0) ? x : a * x;
                break;
            case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                out[i] = std::log(static_cast<T>(1) + std::exp(x));
                break;
            case ActivationLayerInfo::ActivationFunction::SQRT:
                out[i] = std::sqrt(x);
                break;
            case ActivationLayerInfo::ActivationFunction::SQUARE:
                out[i] = x * x;
                break;
            case ActivationLayerInfo::ActivationFunction::TANH:
                out[i] = a * std::tanh(b * x);
                break;
            default:
                ARM_COMPUTE_ERROR("Activation function not recognised");
                break;
        }
    }
}

// Activation Layer for fixed point type
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void activation_layer(const Tensor<T> &in, Tensor<T> &out, ActivationLayerInfo act_info)
{
    using namespace fixed_point_arithmetic;
    int                                     fixed_point_position = in.fixed_point_position();
    ActivationLayerInfo::ActivationFunction act_func             = act_info.activation();
    const fixed_point<T>                    a(act_info.a(), fixed_point_position);
    const fixed_point<T>                    b(act_info.b(), fixed_point_position);
    const fixed_point<T>                    const_0(0, fixed_point_position);
    const fixed_point<T>                    const_1(1, fixed_point_position);

    for(int i = 0; i < in.num_elements(); ++i)
    {
        fixed_point<T> x(in[i], fixed_point_position, true);
        switch(act_func)
        {
            case ActivationLayerInfo::ActivationFunction::ABS:
                out[i] = abs(x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::LINEAR:
                out[i] = add(b, mul(a, x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                out[i] = (const_1 / (const_1 + exp(-x))).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::RELU:
                out[i] = max(const_0, x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                out[i] = min(a, max(const_0, x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                out[i] = (x > const_0) ? x.raw() : mul(a, x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                out[i] = log(const_1 + exp(x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::SQRT:
                out[i] = (const_1 / inv_sqrt(x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::SQUARE:
                out[i] = mul(x, x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::TANH:
                out[i] = mul(a, tanh(mul(b, x))).raw();
                break;
            default:
                ARM_COMPUTE_ERROR("Activation function not recognised");
                break;
        }
    }
}

// Batch Normalization Layer for fixed point type
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void batch_normalization_layer(const Tensor<T> &in, Tensor<T> &out, const Tensor<T> &mean, const Tensor<T> &var, const Tensor<T> &beta, const Tensor<T> &gamma, float epsilon, int fixed_point_position)
{
    const int cols       = static_cast<int>(in.shape()[0]);
    const int rows       = static_cast<int>(in.shape()[1]);
    const int depth      = static_cast<int>(in.shape()[2]);
    int       upper_dims = in.shape().total_size() / (cols * rows * depth);

    for(int r = 0; r < upper_dims; ++r)
    {
        for(int i = 0; i < depth; ++i)
        {
            for(int k = 0; k < rows; ++k)
            {
                for(int l = 0; l < cols; ++l)
                {
                    const int                              pos = l + k * cols + i * rows * cols + r * cols * rows * depth;
                    fixed_point_arithmetic::fixed_point<T> in_qs(in[pos], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> var_qs(var[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> mean_qs(mean[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> beta_qs(beta[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> gamma_qs(gamma[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> epsilon_qs(epsilon, fixed_point_position);

                    auto denominator = fixed_point_arithmetic::inv_sqrt(var_qs + epsilon_qs);
                    auto numerator   = in_qs - mean_qs;
                    auto x_bar       = numerator * denominator;
                    x_bar            = beta_qs + x_bar * gamma_qs;
                    out[pos]         = x_bar.raw();
                }
            }
        }
    }
}

// Batch Normalization Layer for floating point type
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void batch_normalization_layer(const Tensor<T> &in, Tensor<T> &out, const Tensor<T> &mean, const Tensor<T> &var, const Tensor<T> &beta, const Tensor<T> &gamma, float epsilon, int fixed_point_position)
{
    const int cols       = static_cast<int>(in.shape()[0]);
    const int rows       = static_cast<int>(in.shape()[1]);
    const int depth      = static_cast<int>(in.shape()[2]);
    int       upper_dims = in.shape().total_size() / (cols * rows * depth);

    for(int r = 0; r < upper_dims; ++r)
    {
        for(int i = 0; i < depth; ++i)
        {
            for(int k = 0; k < rows; ++k)
            {
                for(int l = 0; l < cols; ++l)
                {
                    const int   pos         = l + k * cols + i * rows * cols + r * cols * rows * depth;
                    const float denominator = sqrt(var[i] + epsilon);
                    const float numerator   = in[pos] - mean[i];
                    const float x_bar       = numerator / denominator;
                    out[pos]                = beta[i] + x_bar * gamma[i];
                }
            }
        }
    }
}

// Depth Concatenate layer
template <typename T>
void depth_concatenate_layer(const std::vector<const Tensor<T> *> &srcs, Tensor<T> &out)
{
    unsigned  depth_offset = 0;
    const int width_out    = out.shape().x();
    const int height_out   = out.shape().y();
    const int depth_out    = out.shape().z();
    const int out_stride_z = width_out * height_out;
    const int batches      = out.shape().total_size_upper(3);

    // Set output tensor to 0
    memset(out.data(), 0, out.num_elements() * element_size_from_data_type(out.data_type()));

    for(unsigned int i = 0; i < srcs.size(); ++i)
    {
        ARM_COMPUTE_ERROR_ON(srcs[i] == nullptr);
        ARM_COMPUTE_ERROR_ON(srcs[i]->data_type() != out.data_type());
        ARM_COMPUTE_ERROR_ON(depth_offset >= out.shape().z());
        ARM_COMPUTE_ERROR_ON(batches != static_cast<int>(srcs[i]->shape().total_size_upper(3)));

        const Tensor<T>   *src    = srcs[i];
        const int          width  = src->shape().x();
        const int          height = src->shape().y();
        const int          depth  = src->shape().z();
        const unsigned int x_diff = (width_out - width) / 2;
        const unsigned int y_diff = (height_out - height) / 2;

        const T *src_ptr = src->data();
        for(int b = 0; b < batches; ++b)
        {
            const unsigned int offset_to_first_element = b * out_stride_z * depth_out + depth_offset * out_stride_z
                                                         + y_diff * width_out + x_diff;
            for(int d = 0; d < depth; ++d)
            {
                for(int r = 0; r < height; ++r)
                {
                    std::copy(src_ptr, src_ptr + width, out.data() + offset_to_first_element + d * out_stride_z + r * width_out);
                    src_ptr += width;
                }
            }
        }

        depth_offset += depth;
    }
}

// Convolution layer
template <typename T>
void convolution_layer(const Tensor<T> &in, const Tensor<T> &weights, const Tensor<T> &bias, Tensor<T> &out, const PadStrideInfo &conv_info)
{
    const int width_in       = in.shape().x();
    const int height_in      = in.shape().y();
    const int depth_in       = in.shape().z();
    const int width_out      = out.shape().x();
    const int height_out     = out.shape().y();
    const int depth_out      = out.shape().z();
    const int width_weights  = weights.shape().x();
    const int height_weights = weights.shape().y();
    const int depth_weights  = weights.shape().z();
    const int pad_xi         = std::min(static_cast<int>(conv_info.pad().first), width_weights / 2);
    const int pad_yi         = std::min(static_cast<int>(conv_info.pad().second), height_weights / 2);
    const int start_xi       = width_weights / 2 - pad_xi;
    const int start_yi       = height_weights / 2 - pad_yi;
    const int end_xi         = width_in - start_xi;
    const int end_yi         = height_in - start_yi;
    const int stride_xi      = conv_info.stride().first;
    const int stride_yi      = conv_info.stride().second;
    const int num_batches    = in.shape().total_size() / (width_in * height_in * depth_in);

    for(int r = 0; r < num_batches; ++r)
    {
        for(int yi = start_yi; yi < end_yi; yi += stride_yi)
        {
            for(int xi = start_xi; xi < end_xi; xi += stride_xi)
            {
                for(int ofm = 0; ofm < depth_out; ++ofm)
                {
                    // Compute input and output offsets
                    const int offset_in  = r * width_in * height_in * depth_in;
                    const int xo         = (xi - start_xi) / stride_xi;
                    const int yo         = (yi - start_yi) / stride_yi;
                    const int offset_out = xo + yo * width_out + ofm * width_out * height_out + r * width_out * height_out * depth_out;

                    // Compute 3D convolution
                    convolution3d(in.data() + offset_in,
                                  weights.data() + ofm * width_weights * height_weights * depth_weights,
                                  bias.data() + ofm,
                                  out.data() + offset_out,
                                  xi, yi,
                                  width_in, height_in, depth_in,
                                  width_weights, height_weights,
                                  static_cast<int8_t>(in.fixed_point_position()));
                }
            }
        }
    }
}

// Fully connected layer
template <typename T>
void fully_connected_layer(const Tensor<T> &in, const Tensor<T> &weights, const Tensor<T> &bias, Tensor<T> &out)
{
    ARM_COMPUTE_ERROR_ON(weights.shape().x() != out.shape().x());
    ARM_COMPUTE_ERROR_ON(weights.shape().y() != in.shape().x() * in.shape().y() * in.shape().z());
    const int cols_weights = weights.shape().x();
    const int rows_weights = weights.shape().y();
    const int num_batches  = in.shape().total_size() / rows_weights;

    for(int k = 0; k < num_batches; ++k)
    {
        vector_matrix_multiply<T>(in.data() + k * rows_weights,
                                  weights.data(),
                                  bias.data(),
                                  out.data() + k * cols_weights,
                                  cols_weights,
                                  rows_weights,
                                  in.fixed_point_position());
    }
}

// Normalization Layer for floating point type
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void normalization_layer(const Tensor<T> &in, Tensor<T> &out, NormalizationLayerInfo norm_info)
{
    const uint32_t norm_size = norm_info.norm_size();
    NormType       type      = norm_info.type();
    float          beta      = norm_info.beta();
    uint32_t       kappa     = norm_info.kappa();

    const int cols       = static_cast<int>(in.shape()[0]);
    const int rows       = static_cast<int>(in.shape()[1]);
    const int depth      = static_cast<int>(in.shape()[2]);
    int       upper_dims = in.shape().total_size() / (cols * rows);

    float coeff       = norm_info.scale_coeff();
    int   radius_cols = norm_size / 2;
    // IN_MAP_1D and CROSS_MAP normalize over a single axis only
    int radius_rows = (NormType::IN_MAP_2D == type) ? norm_size / 2 : 0;

    if(type == NormType::CROSS_MAP)
    {
        // Remove also depth from upper dimensions since it is the axes we want
        // to use for normalization
        upper_dims /= depth;
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int i = 0; i < rows; ++i)
            {
                for(int k = 0; k < cols; ++k)
                {
                    for(int l = 0; l < depth; ++l)
                    {
                        float accumulated_scale = 0.f;
                        for(int j = -radius_cols; j <= radius_cols; ++j)
                        {
                            const int z = l + j;
                            if(z >= 0 && z < depth)
                            {
                                const T value = in[k + i * cols + z * rows * cols + r * cols * rows * depth];
                                accumulated_scale += value * value;
                            }
                        }
                        out[k + i * cols + l * rows * cols + r * cols * rows * depth] = kappa + accumulated_scale * coeff;
                    }
                }
            }
        }
    }
    else
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int i = 0; i < rows; ++i)
            {
                for(int k = 0; k < cols; ++k)
                {
                    float accumulated_scale = 0.f;
                    for(int j = -radius_rows; j <= radius_rows; ++j)
                    {
                        const int y = i + j;
                        for(int l = -radius_cols; l <= radius_cols; ++l)
                        {
                            const int x = k + l;
                            if((x >= 0 && y >= 0) && (x < cols && y < rows))
                            {
                                const T value = in[x + y * cols + r * cols * rows];
                                accumulated_scale += value * value;
                            }
                        }
                    }
                    out[k + i * cols + r * cols * rows] = kappa + accumulated_scale * coeff;
                }
            }
        }
    }

    if(beta == 1.f)
    {
        for(int i = 0; i < out.num_elements(); ++i)
        {
            out[i] = in[i] / out[i];
        }
    }
    else if(beta == 0.5f)
    {
        for(int i = 0; i < out.num_elements(); ++i)
        {
            out[i] = in[i] / std::sqrt(out[i]);
        }
    }
    else
    {
        for(int i = 0; i < out.num_elements(); ++i)
        {
            out[i] = in[i] * std::exp(std::log(out[i]) * -beta);
        }
    }
}
// Normalization Layer for fixed-point types
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void normalization_layer(const Tensor<T> &in, Tensor<T> &out, NormalizationLayerInfo norm_info)
{
    using namespace fixed_point_arithmetic;

    const int fixed_point_position = in.fixed_point_position();

    const uint32_t norm_size = norm_info.norm_size();
    NormType       type      = norm_info.type();
    fixed_point<T> beta(norm_info.beta(), fixed_point_position);
    fixed_point<T> kappa(norm_info.kappa(), fixed_point_position);

    const int cols       = static_cast<int>(in.shape()[0]);
    const int rows       = static_cast<int>(in.shape()[1]);
    const int depth      = static_cast<int>(in.shape()[2]);
    int       upper_dims = in.shape().total_size() / (cols * rows);

    fixed_point<T> coeff(norm_info.scale_coeff(), fixed_point_position);
    int            radius_cols = norm_size / 2;
    // IN_MAP_1D and CROSS_MAP normalize over a single axis only
    int radius_rows = (NormType::IN_MAP_2D == type) ? norm_size / 2 : 0;

    if(type == NormType::CROSS_MAP)
    {
        // Remove also depth from upper dimensions since it is the axes we want
        // to use for normalization
        upper_dims /= depth;
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int i = 0; i < rows; ++i)
            {
                for(int k = 0; k < cols; ++k)
                {
                    for(int l = 0; l < depth; ++l)
                    {
                        fixed_point<T> accumulated_scale(0.f, fixed_point_position);
                        for(int j = -radius_cols; j <= radius_cols; ++j)
                        {
                            const int z = l + j;
                            if(z >= 0 && z < depth)
                            {
                                const T              value = in[k + i * cols + z * rows * cols + r * cols * rows * depth];
                                const fixed_point<T> fp_value(value, fixed_point_position, true);
                                accumulated_scale = add(accumulated_scale, mul(fp_value, fp_value));
                            }
                        }
                        accumulated_scale                                             = add(kappa, mul(accumulated_scale, coeff));
                        out[k + i * cols + l * rows * cols + r * cols * rows * depth] = accumulated_scale.raw();
                    }
                }
            }
        }
    }
    else
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int i = 0; i < rows; ++i)
            {
                for(int k = 0; k < cols; ++k)
                {
                    fixed_point<T> accumulated_scale(0.f, fixed_point_position);
                    for(int j = -radius_rows; j <= radius_rows; ++j)
                    {
                        const int y = i + j;
                        for(int l = -radius_cols; l <= radius_cols; ++l)
                        {
                            const int x = k + l;
                            if((x >= 0 && y >= 0) && (x < cols && y < rows))
                            {
                                const T              value = in[x + y * cols + r * cols * rows];
                                const fixed_point<T> fp_value(value, fixed_point_position, true);
                                accumulated_scale = add(accumulated_scale, mul(fp_value, fp_value));
                            }
                        }
                    }
                    accumulated_scale                   = add(kappa, mul(accumulated_scale, coeff));
                    out[k + i * cols + r * cols * rows] = accumulated_scale.raw();
                }
            }
        }
    }

    if(norm_info.beta() == 1.f)
    {
        for(int i = 0; i < out.num_elements(); ++i)
        {
            fixed_point<T> res = div(fixed_point<T>(in[i], fixed_point_position, true), fixed_point<T>(out[i], fixed_point_position, true));
            out[i]             = res.raw();
        }
    }
    else
    {
        const fixed_point<T> beta(norm_info.beta(), fixed_point_position);
        for(int i = 0; i < out.num_elements(); ++i)
        {
            fixed_point<T> res = pow(fixed_point<T>(out[i], fixed_point_position, true), beta);
            res                = div(fixed_point<T>(in[i], fixed_point_position, true), res);
            out[i]             = res.raw();
        }
    }
}

// Pooling layer
template <typename T>
void pooling_layer(const Tensor<T> &in, Tensor<T> &out, PoolingLayerInfo pool_info, int fixed_point_position)
{
    const int   pool_size     = pool_info.pool_size();
    PoolingType type          = pool_info.pool_type();
    int         pool_stride_x = 0;
    int         pool_stride_y = 0;
    int         pad_x         = 0;
    int         pad_y         = 0;
    std::tie(pool_stride_x, pool_stride_y) = pool_info.pad_stride_info().stride();
    std::tie(pad_x, pad_y)                 = pool_info.pad_stride_info().pad();

    const int w_in = static_cast<int>(in.shape()[0]);
    const int h_in = static_cast<int>(in.shape()[1]);

    const int w_out = static_cast<int>(out.shape()[0]);
    const int h_out = static_cast<int>(out.shape()[1]);

    int upper_dims = in.shape().total_size() / (w_in * h_in);

    int pooled_w = 0;
    int pooled_h = 0;
    if(pool_info.pad_stride_info().round() == DimensionRoundingType::CEIL)
    {
        pooled_w = static_cast<int>(ceil(static_cast<float>(w_in + 2 * pad_x - pool_size) / pool_stride_x)) + 1;
        pooled_h = static_cast<int>(ceil(static_cast<float>(h_in + 2 * pad_y - pool_size) / pool_stride_y)) + 1;
    }
    else
    {
        pooled_w = static_cast<int>(floor(static_cast<float>(w_in + 2 * pad_x - pool_size) / pool_stride_x)) + 1;
        pooled_h = static_cast<int>(floor(static_cast<float>(h_in + 2 * pad_y - pool_size) / pool_stride_y)) + 1;
    }

    if((pooled_w - 1) * pool_stride_x >= w_in + pad_x)
    {
        --pooled_w;
    }
    if((pooled_h - 1) * pool_stride_y >= h_in + pad_y)
    {
        --pooled_h;
    }

    if(type == PoolingType::MAX)
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < pooled_h; ++h)
            {
                for(int w = 0; w < pooled_w; ++w)
                {
                    int wstart = w * pool_stride_x - pad_x;
                    int hstart = h * pool_stride_y - pad_y;
                    int wend   = std::min(wstart + pool_size, w_in);
                    int hend   = std::min(hstart + pool_size, h_in);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);

                    T max_val = std::numeric_limits<T>::lowest();
                    for(int y = hstart; y < hend; ++y)
                    {
                        for(int x = wstart; x < wend; ++x)
                        {
                            const T val = in[r * h_in * w_in + y * w_in + x];
                            if(val > max_val)
                            {
                                max_val = val;
                            }
                        }
                    }

                    out[r * h_out * w_out + h * pooled_w + w] = max_val;
                }
            }
        }
    }
    else // Average pooling
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < pooled_h; ++h)
            {
                for(int w = 0; w < pooled_w; ++w)
                {
                    T   avg_val = 0;
                    int wstart  = w * pool_stride_x - pad_x;
                    int hstart  = h * pool_stride_y - pad_y;
                    int wend    = std::min(wstart + pool_size, w_in + pad_x);
                    int hend    = std::min(hstart + pool_size, h_in + pad_y);
                    int pool    = (hend - hstart) * (wend - wstart);
                    wstart      = std::max(wstart, 0);
                    hstart      = std::max(hstart, 0);
                    wend        = std::min(wend, w_in);
                    hend        = std::min(hend, h_in);
                    if(is_floating_point<T>::value)
                    {
                        for(int y = hstart; y < hend; ++y)
                        {
                            for(int x = wstart; x < wend; ++x)
                            {
                                avg_val += in[r * h_in * w_in + y * w_in + x];
                            }
                        }
                        out[r * h_out * w_out + h * pooled_w + w] = avg_val / pool;
                    }
                    else
                    {
                        static std::array<qint8_t, 10> scale_values_q8 =
                        { { 0x0, 0x0, 0x40, 0x2A, 0x20, 0x19, 0x15, 0x12, 0x10, 0xE } };

                        for(int y = hstart; y < hend; ++y)
                        {
                            for(int x = wstart; x < wend; ++x)
                            {
                                avg_val = sqadd_qs8(avg_val, in[r * h_in * w_in + y * w_in + x]);
                            }
                        }
                        out[r * h_out * w_out + h * pooled_w + w] = sqmul_qs8(avg_val, (scale_values_q8[pool] >> (7 - fixed_point_position)), fixed_point_position);
                    }
                }
            }
        }
    }
}

// Pooling layer
template <typename T>
void roi_pooling_layer(const Tensor<T> &in, Tensor<T> &out, const std::vector<ROI> &rois, const ROIPoolingLayerInfo &pool_info)
{
    const int   num_rois   = rois.size();
    const int   width_in   = in.shape().x();
    const int   height_in  = in.shape().y();
    const int   fms        = in.shape().z();
    const int   volume_in  = width_in * height_in * fms;
    const int   pool_w     = pool_info.pooled_width();
    const int   pool_h     = pool_info.pooled_height();
    const int   volume_out = pool_w * pool_h * fms;
    const float roi_scale  = pool_info.spatial_scale();

    // Iterate through all rois
    for(int roi_idx = 0; roi_idx < num_rois; ++roi_idx)
    {
        // Get dimensions of current ROI
        const ROI &roi = rois[roi_idx];

        int batch_id    = roi.batch_idx;
        int roi_start_x = support::cpp11::round(roi.rect.x * roi_scale);
        int roi_start_y = support::cpp11::round(roi.rect.y * roi_scale);
        int roi_width   = std::max(support::cpp11::round(roi.rect.width * roi_scale), 1.f);
        int roi_height  = std::max(support::cpp11::round(roi.rect.height * roi_scale), 1.f);

        // Determine pooling regions
        float pool_region_size_x = static_cast<float>(roi_width) / pool_w;
        float pool_region_size_y = static_cast<float>(roi_height) / pool_h;

        // Iterate through all channel
        for(int fm = 0; fm < fms; ++fm)
        {
            // Calculate each output pixel
            for(int py = 0; py < pool_h; ++py)
            {
                for(int px = 0; px < pool_w; ++px)
                {
                    int region_start_x = static_cast<int>(std::floor(px * pool_region_size_x));
                    int region_end_x   = static_cast<int>(std::ceil((px + 1) * pool_region_size_x));
                    int region_start_y = static_cast<int>(std::floor(py * pool_region_size_y));
                    int region_end_y   = static_cast<int>(std::ceil((py + 1) * pool_region_size_y));

                    region_start_x = std::min(std::max(region_start_x + roi_start_x, 0), width_in);
                    region_end_x   = std::min(std::max(region_end_x + roi_start_x, 0), width_in);
                    region_start_y = std::min(std::max(region_start_y + roi_start_y, 0), height_in);
                    region_end_y   = std::min(std::max(region_end_y + roi_start_y, 0), height_in);

                    // Iterate through each pixel in the pooling region
                    if((region_end_x <= region_start_x) || (region_end_y <= region_start_y))
                    {
                        out[roi_idx * volume_out + fm * pool_w * pool_h + py * pool_w + px] = 0;
                    }
                    else
                    {
                        T curr_max = std::numeric_limits<T>::lowest();
                        for(int j = region_start_y; j < region_end_y; ++j)
                        {
                            for(int i = region_start_x; i < region_end_x; ++i)
                            {
                                const auto val = in[batch_id * volume_in + fm * width_in * height_in + j * width_in + i];
                                curr_max       = std::max(val, curr_max);
                            }
                        }
                        out[roi_idx * volume_out + fm * pool_w * pool_h + py * pool_w + px] = curr_max;
                    }
                }
            }
        }
    }
}

// Softmax Layer
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void softmax_layer(const Tensor<T> &in, Tensor<T> &out)
{
    const int cols       = static_cast<int>(in.shape()[0]);
    const int upper_dims = in.shape().total_size() / cols;
    for(int r = 0; r < upper_dims; ++r)
    {
        // Find max
        T max = std::numeric_limits<T>::lowest();
        for(int c = 0; c < cols; ++c)
        {
            const T x = in[r * cols + c];
            if(x > max)
            {
                max = x;
            }
        }

        // Regularize
        T sum = 0;
        for(int c = 0; c < cols; ++c)
        {
            const T res       = exp(in[r * cols + c] - max);
            out[r * cols + c] = res;
            sum += res;
        }

        // Normalize
        const T norm_val = 1 / sum;
        for(int c = 0; c < cols; ++c)
        {
            out[r * cols + c] *= norm_val;
        }
    }
}
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void softmax_layer(const Tensor<T> &in, Tensor<T> &out)
{
    using namespace fixed_point_arithmetic;
    using promoted_T = typename test::traits::promote<T>::type;

    const int fixed_point_position = in.fixed_point_position();
    const int cols                 = static_cast<int>(in.shape()[0]);
    const int upper_dims           = in.shape().total_size() / cols;

    for(int r = 0; r < upper_dims; ++r)
    {
        // Find max
        fixed_point<T> max(std::numeric_limits<T>::lowest(), fixed_point_position, true);
        for(int c = 0; c < cols; ++c)
        {
            const fixed_point<T> x(in[r * cols + c], fixed_point_position, true);
            if(x > max)
            {
                max = x;
            }
        }

        // Regularize
        fixed_point<promoted_T> sum(0, fixed_point_position);
        for(int c = 0; c < cols; ++c)
        {
            const fixed_point<T> x(in[r * cols + c], fixed_point_position, true);
            fixed_point<T>       res = exp(x - max);
            out[r * cols + c]        = res.raw();
            sum                      = add(sum, static_cast<fixed_point<promoted_T>>(res));
        }

        // Normalize
        fixed_point<T> sat_sum(sum);
        for(int c = 0; c < cols; ++c)
        {
            const fixed_point<T> x(out[r * cols + c], fixed_point_position, true);
            out[r * cols + c] = div(x, sat_sum).raw();
        }
    }
}

// Fixed point operations
template <typename T>
void fixed_point_operation(const Tensor<T> &in, Tensor<T> &out, FixedPointOp op)
{
    int p = in.fixed_point_position();
    switch(op)
    {
        case FixedPointOp::EXP:
            for(int i = 0; i < in.num_elements(); ++i)
            {
                out[i] = fixed_point_arithmetic::exp(fixed_point_arithmetic::fixed_point<T>(in[i], p, true)).raw();
            }
            break;
        case FixedPointOp::LOG:
            for(int i = 0; i < in.num_elements(); ++i)
            {
                out[i] = fixed_point_arithmetic::log(fixed_point_arithmetic::fixed_point<T>(in[i], p, true)).raw();
            }
            break;
        case FixedPointOp::INV_SQRT:
            for(int i = 0; i < in.num_elements(); ++i)
            {
                out[i] = fixed_point_arithmetic::inv_sqrt(fixed_point_arithmetic::fixed_point<T>(in[i], p, true)).raw();
            }
            break;
        case FixedPointOp::RECIPROCAL:
            for(int i = 0; i < in.num_elements(); ++i)
            {
                out[i] = fixed_point_arithmetic::div(fixed_point_arithmetic::fixed_point<T>(1, p), fixed_point_arithmetic::fixed_point<T>(in[i], p, true)).raw();
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Fixed point operation not supported");
            break;
    }
}

// Tensor print
template <typename T>
void print(const Tensor<T> &in, std::ostream &out)
{
    out << "\n";
    for(int i = 0; i < in.num_elements(); ++i)
    {
        out << in[i] << " ";
    }
    out << "\n";
}
} // namespace tensor_operations
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* __ARM_COMPUTE_TEST_TENSOR_OPERATIONS_H__ */
