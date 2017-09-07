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

#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"
#include "tests/Types.h"
#include "tests/Utils.h"
#include "tests/validation_old/FixedPoint.h"
#include "tests/validation_old/Tensor.h"
#include "tests/validation_old/ValidationUserConfiguration.h"
#include "tests/validation_old/half.h"

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

        // Iterate through all channel
        for(int fm = 0; fm < fms; ++fm)
        {
            // Calculate each output pixel
            for(int py = 0; py < pool_h; ++py)
            {
                for(int px = 0; px < pool_w; ++px)
                {
                    int region_start_x = static_cast<int>(std::floor((static_cast<float>(px) / pool_w) * roi_width));
                    int region_end_x   = static_cast<int>(std::floor((static_cast<float>(px + 1) / pool_w) * roi_width));
                    int region_start_y = static_cast<int>(std::floor((static_cast<float>(py) / pool_h) * roi_height));
                    int region_end_y   = static_cast<int>(std::floor((static_cast<float>(py + 1) / pool_h) * roi_height));

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
