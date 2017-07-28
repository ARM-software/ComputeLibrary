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

#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"
#include "tests/Types.h"
#include "tests/Utils.h"
#include "tests/validation/FixedPoint.h"
#include "tests/validation/Tensor.h"
#include "tests/validation/ValidationUserConfiguration.h"
#include "tests/validation/half.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

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
      std::is_same<float, typename std::remove_cv<T>::type>::value || std::is_same<half_float::half, typename std::remove_cv<T>::type>::value
      || std::is_same<double, typename std::remove_cv<T>::type>::value || std::is_same<long double, typename std::remove_cv<T>::type>::value >
{
};

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void vector_matrix_multiply(const T *in, const T *weights, const T *bias, T *out, int cols_weights, int rows_weights, uint8_t fixed_point_position)
{
    for(int x = 0; x < cols_weights; ++x)
    {
        T acc(0);
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
template <typename T>
T tensor_elem_at(const Tensor<T> &in, Coordinates coord, BorderMode border_mode, T constant_border_value)
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
        }
        else
        {
            return constant_border_value;
        }
    }

    return in[coord2index(in.shape(), coord)];
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

// Sobel 7x7
template <typename T1, typename T2>
void sobel_7x7(Tensor<T1> &in, Tensor<T2> &out_x, Tensor<T2> &out_y, BorderMode border_mode, uint8_t constant_border_value)
{
    const std::array<int8_t, 49> sobel_x{ {
            -1, -4, -5, 0, 5, 4, 1,
            -6, -24, -30, 0, 30, 24, 6,
            -15, -60, -75, 0, 75, 60, 15,
            -20, -80, -100, 0, 100, 80, 20,
            -15, -60, -75, 0, 75, 60, 15,
            -6, -24, -30, 0, 30, 24, 6,
            -1, -4, -5, 0, 5, 4, 1
        } };

    const std::array<int8_t, 49> sobel_y{ {
            -1, -6, -15, -20, -15, -6, -1,
            -4, -24, -60, -80, -60, -24, -4,
            -5, -30, -75, -100, -75, -30, -5,
            0, 0, 0, 0, 0, 0, 0,
            5, 30, 75, 100, 75, 30, 5,
            4, 24, 60, 80, 60, 24, 4,
            1, 6, 15, 20, 15, 6, 1
        } };

    for(int element_idx = 0; element_idx < in.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(in.shape(), element_idx);

        apply_2d_spatial_filter(id, in, out_x, TensorShape(7U, 7U), sobel_x.data(), 1.f, border_mode, constant_border_value);
        apply_2d_spatial_filter(id, in, out_y, TensorShape(7U, 7U), sobel_y.data(), 1.f, border_mode, constant_border_value);
    }
}

template <typename T>
void non_maxima_suppression_3x3(Tensor<T> &in, Tensor<T> &out, BorderMode border_mode)
{
    for(int i = 0; i < in.num_elements(); ++i)
    {
        Coordinates coord = index2coord(in.shape(), i);
        int         x     = coord.x();
        int         y     = coord.y();

        if(in[i] >= tensor_elem_at(in, Coordinates(x - 1, y - 1), border_mode, 0.f) && in[i] >= tensor_elem_at(in, Coordinates(x, y - 1), border_mode, 0.f)
           && in[i] >= tensor_elem_at(in, Coordinates(x + 1, y - 1), border_mode, 0.f) && in[i] >= tensor_elem_at(in, Coordinates(x - 1, y), border_mode, 0.f)
           && in[i] > tensor_elem_at(in, Coordinates(x + 1, y), border_mode, 0.f) && in[i] > tensor_elem_at(in, Coordinates(x - 1, y + 1), border_mode, 0.f)
           && in[i] > tensor_elem_at(in, Coordinates(x, y + 1), border_mode, 0.f) && in[i] > tensor_elem_at(in, Coordinates(x + 1, y + 1), border_mode, 0.f))
        {
            out[i] = in[i];
        }
        else
        {
            out[i] = 0;
        }
    }
}

// Harris corners
template <typename T1, typename T2, typename T3>
void harris_corners(Tensor<T1> &in, Tensor<T2> &Gx, Tensor<T2> &Gy, Tensor<T3> &candidates, Tensor<T3> &non_maxima, float threshold, float min_dist, float sensitivity,
                    int32_t gradient_size, int32_t block_size, KeyPointArray &corners, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(block_size != 3 && block_size != 5 && block_size != 7);

    ValidRegion valid_region = shape_to_valid_region(candidates.shape());
    float       norm_factor  = 0.f;

    // Sobel
    switch(gradient_size)
    {
        case 3:
            sobel_3x3(in, Gx, Gy, border_mode, constant_border_value);
            norm_factor = 1.f / (4 * 255 * block_size);
            break;
        case 5:
            sobel_5x5(in, Gx, Gy, border_mode, constant_border_value);
            norm_factor = 1.f / (16 * 255 * block_size);
            break;
        case 7:
            sobel_7x7(in, Gx, Gy, border_mode, constant_border_value);
            norm_factor = 1.f / (64 * 255 * block_size);
            break;
        default:
            ARM_COMPUTE_ERROR("Gradient size not supported.");
    }

    //Calculate scores
    for(int i = 0; i < in.num_elements(); ++i)
    {
        Coordinates in_coord = index2coord(in.shape(), i);

        float Gx2 = 0;
        float Gy2 = 0;
        float Gxy = 0;

        // Calculate Gx^2, Gy^2 and Gxy within the given window
        for(int y = in_coord.y() - block_size / 2; y <= in_coord.y() + block_size / 2; ++y)
        {
            for(int x = in_coord.x() - block_size / 2; x <= in_coord.x() + block_size / 2; ++x)
            {
                Coordinates block_coord(x, y);

                float norm_gx = tensor_elem_at(Gx, block_coord, border_mode, static_cast<T2>(constant_border_value)) * norm_factor;
                float norm_gy = tensor_elem_at(Gy, block_coord, border_mode, static_cast<T2>(constant_border_value)) * norm_factor;

                Gx2 += std::pow(norm_gx, 2);
                Gy2 += std::pow(norm_gy, 2);
                Gxy += norm_gx * norm_gy;
            }
        }

        float trace2   = std::pow(Gx2 + Gy2, 2);
        float det      = Gx2 * Gy2 - std::pow(Gxy, 2);
        float response = det - sensitivity * trace2;

        if(response > threshold)
        {
            candidates[i] = response;
        }
        else
        {
            candidates[i] = 0.f;
        }
    }

    // Update valid region and remove candidates on borders for border_mode == UNDEFINED
    if(border_mode == BorderMode::UNDEFINED)
    {
        valid_region = shape_to_valid_region(candidates.shape(), true, BorderSize((gradient_size / 2) + (block_size / 2)));

        for(int i = 0; i < candidates.num_elements(); ++i)
        {
            if(!is_in_valid_region(valid_region, index2coord(candidates.shape(), i)))
            {
                candidates[i] = 0.f;
            }
        }
    }

    // Suppress non-maxima candidates
    non_maxima_suppression_3x3(candidates, non_maxima, border_mode != BorderMode::UNDEFINED ? BorderMode::CONSTANT : BorderMode::UNDEFINED);
    if(border_mode == BorderMode::UNDEFINED)
    {
        valid_region = shape_to_valid_region(non_maxima.shape(), true, BorderSize((gradient_size / 2) + (block_size / 2) + 1));
    }

    // Create vector of candidate corners
    KeyPointArray candidates_vector(corners.max_num_values());
    for(int i = 0; i < non_maxima.num_elements(); ++i)
    {
        Coordinates coord = index2coord(non_maxima.shape(), i);

        if(non_maxima[i] != 0.f && is_in_valid_region(valid_region, coord))
        {
            KeyPoint corner;
            corner.x               = coord.x();
            corner.y               = coord.y();
            corner.tracking_status = 1;
            corner.strength        = non_maxima[i];

            corner.scale       = 0.f;
            corner.orientation = 0.f;
            corner.error       = 0.f;

            candidates_vector.push_back(corner);
        }
    }

    // If there are any candidates, sort them by strength and add them to the output corners vector if there are no stronger corners within the given euclidean radius
    if(candidates_vector.num_values() > 0)
    {
        std::sort(candidates_vector.buffer(), candidates_vector.buffer() + candidates_vector.num_values(), [](KeyPoint a, KeyPoint b)
        {
            return a.strength > b.strength;
        });
        corners.push_back(candidates_vector.at(0));

        for(size_t j = 0; j < candidates_vector.num_values(); ++j)
        {
            bool    found = false;
            int32_t x     = candidates_vector.at(j).x;
            int32_t y     = candidates_vector.at(j).y;

            for(size_t i = 0; i < corners.num_values(); ++i)
            {
                int32_t corners_x = corners.at(i).x;
                int32_t corners_y = corners.at(i).y;

                // Euclidean distance
                if(std::sqrt((std::pow(x - corners_x, 2) + std::pow(y - corners_y, 2))) < min_dist)
                {
                    found = true;
                }
            }

            // If no stronger corners within the given euclidean radius
            if(!found)
            {
                corners.push_back(candidates_vector.at(j));
            }
        }
    }
}

template <typename T>
void compute_min_max(const Tensor<T> &in, void *min, void *max)
{
    using type = typename std::conditional<std::is_same<T, float>::value, float, int32_t>::type;

    // Set min and max to first pixel
    type tmp_min = static_cast<type>(in[0]);
    type tmp_max = static_cast<type>(in[0]);

    // Look for min and max values
    for(int i = 1; i < in.num_elements(); ++i)
    {
        if(static_cast<type>(in[i]) < tmp_min)
        {
            tmp_min = static_cast<type>(in[i]);
        }
        if(static_cast<type>(in[i]) > tmp_max)
        {
            tmp_max = static_cast<type>(in[i]);
        }
    }

    *static_cast<type *>(min) = tmp_min;
    *static_cast<type *>(max) = tmp_max;
}

// Min max location
template <typename T1>
void min_max_location(const Tensor<T1> &in, void *min, void *max, IArray<Coordinates2D> &min_loc, IArray<Coordinates2D> &max_loc, uint32_t &min_count, uint32_t &max_count)
{
    const size_t width = in.shape().x();

    compute_min_max(in, min, max);

    using type = typename std::conditional<std::is_same<T1, float>::value, float, int32_t>::type;

    type min_value = *static_cast<type *>(min);
    type max_value = *static_cast<type *>(max);

    min_count = 0;
    max_count = 0;
    for(int i = 0; i < in.num_elements(); ++i)
    {
        if(static_cast<type>(in[i]) == min_value)
        {
            Coordinates2D min_coord;
            min_coord.x = static_cast<int32_t>(i % width);
            min_coord.y = static_cast<int32_t>(i / width);

            min_loc.push_back(min_coord);

            min_count++;
        }
        if(static_cast<type>(in[i]) == max_value)
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
        intermediate_type val(std::abs(static_cast<intermediate_type>(in1[i]) - static_cast<intermediate_type>(in2[i])));
        out[i] = saturate_cast<T3>(val);
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

// Accumulate weighted total_size   = init_auto_padding(tensor_shape, num_channels, type);
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
void fixed_point_pixel_wise_multiplication(const Tensor<T> &in1, const Tensor<T> &in2, Tensor<T> &out, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
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

    const fixed_point<T> fp_scale(scale, fixed_point_position);
    const bool           is_sat = convert_policy == ConvertPolicy::SATURATE;

    for(int i = 0; i < in1.num_elements(); ++i)
    {
        const fixed_point<T> val1(in1[i], fixed_point_position, true);
        fixed_point<T>       res(in2[i], fixed_point_position, true);
        if(is_sat)
        {
            res = mul(mul(res, val1), fp_scale);
        }
        else
        {
            res = mul<OverflowPolicy::WRAP>(mul<OverflowPolicy::WRAP>(res, val1), fp_scale);
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

template <typename T>
T bilinear_policy(const Tensor<T> &in, Coordinates id, float xn, float yn, BorderMode border_mode, uint8_t constant_border_value)
{
    int idx = std::floor(xn);
    int idy = std::floor(yn);

    const float dx   = xn - idx;
    const float dy   = yn - idy;
    const float dx_1 = 1.0f - dx;
    const float dy_1 = 1.0f - dy;

    id.set(0, idx);
    id.set(1, idy);
    const T tl = tensor_elem_at(in, id, border_mode, constant_border_value);
    id.set(0, idx + 1);
    id.set(1, idy);
    const T tr = tensor_elem_at(in, id, border_mode, constant_border_value);
    id.set(0, idx);
    id.set(1, idy + 1);
    const T bl = tensor_elem_at(in, id, border_mode, constant_border_value);
    id.set(0, idx + 1);
    id.set(1, idy + 1);
    const T br = tensor_elem_at(in, id, border_mode, constant_border_value);

    return tl * (dx_1 * dy_1) + tr * (dx * dy_1) + bl * (dx_1 * dy) + br * (dx * dy);
}

bool valid_bilinear_policy(float xn, float yn, int width, int height, BorderMode border_mode)
{
    if(border_mode != BorderMode::UNDEFINED)
    {
        return true;
    }
    if((0 <= yn + 1) && (yn + 1 < height) && (0 <= xn + 1) && (xn + 1 < width))
    {
        return true;
    }
    return false;
}

// Warp Perspective
template <typename T>
void warp_perspective(const Tensor<T> &in, Tensor<T> &out, Tensor<T> &valid_mask, const float *matrix, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value)
{
    // x0 = M00 * x + M01 * y + M02
    // y0 = M10 * x + M11 * y + M12
    // z0 = M20 * x + M21 * y + M22
    // xn = x0 / z0
    // yn = y0 / z0
    const float M00 = matrix[0];
    const float M10 = matrix[1];
    const float M20 = matrix[2];
    const float M01 = matrix[0 + 1 * 3];
    const float M11 = matrix[1 + 1 * 3];
    const float M21 = matrix[2 + 1 * 3];
    const float M02 = matrix[0 + 2 * 3];
    const float M12 = matrix[1 + 2 * 3];
    const float M22 = matrix[2 + 2 * 3];

    const int width  = in.shape().x();
    const int height = in.shape().y();

    for(int element_idx = 0; element_idx < in.num_elements(); ++element_idx)
    {
        valid_mask[element_idx] = 1;
        Coordinates id          = index2coord(in.shape(), element_idx);
        int         idx         = id.x();
        int         idy         = id.y();
        const float z0          = M20 * idx + M21 * idy + M22;

        float x0 = (M00 * idx + M01 * idy + M02);
        float y0 = (M10 * idx + M11 * idy + M12);

        float xn = x0 / z0;
        float yn = y0 / z0;
        id.set(0, static_cast<int>(std::floor(xn)));
        id.set(1, static_cast<int>(std::floor(yn)));
        if((0 <= yn) && (yn < height) && (0 <= xn) && (xn < width))
        {
            switch(policy)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    out[element_idx] = tensor_elem_at(in, id, border_mode, constant_border_value);
                    break;
                case InterpolationPolicy::BILINEAR:
                    (valid_bilinear_policy(xn, yn, width, height, border_mode)) ? out[element_idx] = bilinear_policy(in, id, xn, yn, border_mode, constant_border_value) : valid_mask[element_idx] = 0;
                    break;
                case InterpolationPolicy::AREA:
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }
        else
        {
            if(border_mode == BorderMode::UNDEFINED)
            {
                valid_mask[element_idx] = 0;
            }
            else
            {
                switch(policy)
                {
                    case InterpolationPolicy::NEAREST_NEIGHBOR:
                        if(border_mode == BorderMode::CONSTANT)
                        {
                            out[element_idx] = constant_border_value;
                        }
                        else if(border_mode == BorderMode::REPLICATE)
                        {
                            id.set(0, std::max(0, std::min(static_cast<int>(xn), width - 1)));
                            id.set(1, std::max(0, std::min(static_cast<int>(yn), height - 1)));
                            out[element_idx] = in[coord2index(in.shape(), id)];
                        }
                        break;
                    case InterpolationPolicy::BILINEAR:
                        out[element_idx] = bilinear_policy(in, id, xn, yn, border_mode, constant_border_value);
                        break;
                    case InterpolationPolicy::AREA:
                    default:
                        ARM_COMPUTE_ERROR("Interpolation not supported");
                }
            }
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

// Pooling layer
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type * = nullptr>
void pooling_layer(const Tensor<T> &in, Tensor<T> &out, PoolingLayerInfo pool_info)
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
                    T   avg_val(0);
                    int wstart = w * pool_stride_x - pad_x;
                    int hstart = h * pool_stride_y - pad_y;
                    int wend   = std::min(wstart + pool_size, w_in + pad_x);
                    int hend   = std::min(hstart + pool_size, h_in + pad_y);
                    int pool   = (hend - hstart) * (wend - wstart);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);
                    wend       = std::min(wend, w_in);
                    hend       = std::min(hend, h_in);

                    for(int y = hstart; y < hend; ++y)
                    {
                        for(int x = wstart; x < wend; ++x)
                        {
                            avg_val += in[r * h_in * w_in + y * w_in + x];
                        }
                    }
                    out[r * h_out * w_out + h * pooled_w + w] = avg_val / pool;
                }
            }
        }
    }
}

// Pooling layer
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
void pooling_layer(const Tensor<T> &in, Tensor<T> &out, PoolingLayerInfo pool_info)
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
                            T val = in[r * h_in * w_in + y * w_in + x];
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
                    int wstart = w * pool_stride_x - pad_x;
                    int hstart = h * pool_stride_y - pad_y;
                    int wend   = std::min(wstart + pool_size, w_in + pad_x);
                    int hend   = std::min(hstart + pool_size, h_in + pad_y);
                    int pool   = (hend - hstart) * (wend - wstart);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);
                    wend       = std::min(wend, w_in);
                    hend       = std::min(hend, h_in);

                    using namespace fixed_point_arithmetic;

                    const int            fixed_point_position = in.fixed_point_position();
                    const fixed_point<T> invpool_fp(1.f / static_cast<float>(pool), fixed_point_position);
                    fixed_point<T>       avg_val(0, fixed_point_position, true);
                    for(int y = hstart; y < hend; ++y)
                    {
                        for(int x = wstart; x < wend; ++x)
                        {
                            const fixed_point<T> in_fp(in[r * h_in * w_in + y * w_in + x], fixed_point_position, true);
                            avg_val = add(avg_val, in_fp);
                        }
                    }
                    out[r * h_out * w_out + h * pooled_w + w] = mul(avg_val, invpool_fp).raw();
                }
            }
        }
    }
}

// ROI Pooling layer
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
