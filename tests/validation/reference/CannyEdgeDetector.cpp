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
#include "CannyEdgeDetector.h"

#include "Utils.h"
#include "support/ToolchainSupport.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Magnitude.h"
#include "tests/validation/reference/NonMaximaSuppression.h"
#include "tests/validation/reference/Phase.h"
#include "tests/validation/reference/Sobel.h"

#include <cmath>
#include <stack>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
const auto MARK_ZERO  = 0u;
const auto MARK_MAYBE = 127u;
const auto MARK_EDGE  = 255u;

template <typename T>
void trace_edge(SimpleTensor<T> &dst, const ValidRegion &valid_region)
{
    std::stack<Coordinates> pixels_stack;
    for(auto i = 0; i < dst.num_elements(); ++i)
    {
        if(dst[i] == MARK_EDGE)
        {
            pixels_stack.push(index2coord(dst.shape(), i));
        }
    }

    while(!pixels_stack.empty())
    {
        const Coordinates pixel_coord = pixels_stack.top();
        pixels_stack.pop();

        std::array<Coordinates, 8> neighbours =
        {
            {
                Coordinates(pixel_coord.x() - 1, pixel_coord.y() + 0),
                Coordinates(pixel_coord.x() + 1, pixel_coord.y() + 0),
                Coordinates(pixel_coord.x() - 1, pixel_coord.y() - 1),
                Coordinates(pixel_coord.x() + 1, pixel_coord.y() + 1),
                Coordinates(pixel_coord.x() + 0, pixel_coord.y() - 1),
                Coordinates(pixel_coord.x() + 0, pixel_coord.y() + 1),
                Coordinates(pixel_coord.x() + 1, pixel_coord.y() - 1),
                Coordinates(pixel_coord.x() - 1, pixel_coord.y() + 1)
            }
        };

        // Mark MAYBE neighbours as edges since they are next to an EDGE
        std::for_each(neighbours.begin(), neighbours.end(), [&](Coordinates & coord)
        {
            if(is_in_valid_region(valid_region, coord))
            {
                const size_t pixel_index = coord2index(dst.shape(), coord);
                const T      pixel       = dst[pixel_index];
                if(pixel == MARK_MAYBE)
                {
                    dst[pixel_index] = MARK_EDGE;
                    pixels_stack.push(coord);
                }
            }
        });
    }

    // Mark all remaining MAYBE pixels as ZERO (not edges)
    for(auto i = 0; i < dst.num_elements(); ++i)
    {
        if(dst[i] == MARK_MAYBE)
        {
            dst[i] = MARK_ZERO;
        }
    }
}

template <typename U, typename T>
SimpleTensor<T> canny_edge_detector_impl(const SimpleTensor<T> &src, int32_t upper, int32_t lower, int gradient_size, MagnitudeType norm_type,
                                         BorderMode border_mode, T constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(gradient_size != 3 && gradient_size != 5 && gradient_size != 7);
    ARM_COMPUTE_ERROR_ON(lower < 0 || lower >= upper);

    // Output: T == uint8_t
    SimpleTensor<T> dst{ src.shape(), src.data_type() };
    ValidRegion     valid_region = shape_to_valid_region(src.shape(), border_mode == BorderMode::UNDEFINED, BorderSize(gradient_size / 2 + 1));

    // Sobel computation: U == int16_t or int32_t
    SimpleTensor<U> gx{};
    SimpleTensor<U> gy{};
    std::tie(gx, gy) = sobel<U>(src, gradient_size, border_mode, constant_border_value, GradientDimension::GRAD_XY);

    using unsigned_U = typename traits::make_unsigned_conditional_t<U>::type;
    using promoted_U = typename common_promoted_signed_type<U>::intermediate_type;

    // Gradient magnitude and phase (edge direction)
    const DataType           mag_data_type = gx.data_type() == DataType::S16 ? DataType::U16 : DataType::U32;
    SimpleTensor<unsigned_U> grad_mag{ gx.shape(), mag_data_type };
    SimpleTensor<uint8_t>    grad_dir{ gy.shape(), DataType::U8 };

    for(auto i = 0; i < grad_mag.num_elements(); ++i)
    {
        double mag = 0.f;

        if(norm_type == MagnitudeType::L2NORM)
        {
            mag = support::cpp11::round(std::sqrt(static_cast<promoted_U>(gx[i]) * gx[i] + static_cast<promoted_U>(gy[i]) * gy[i]));
        }
        else // MagnitudeType::L1NORM
        {
            mag = static_cast<promoted_U>(std::abs(gx[i])) + static_cast<promoted_U>(std::abs(gy[i]));
        }

        float angle = 180.f * std::atan2(static_cast<float>(gy[i]), static_cast<float>(gx[i])) / M_PI;
        grad_dir[i] = support::cpp11::round(angle < 0.f ? 180 + angle : angle);
        grad_mag[i] = saturate_cast<unsigned_U>(mag);
    }

    /*
        Quantise the phase into 4 directions
          0°  dir=0    0.0 <= p <  22.5 or 157.5 <= p < 180
         45°  dir=1   22.5 <= p <  67.5
         90°  dir=2   67.5 <= p < 112.5
        135°  dir=3  112.5 <= p < 157.5
    */
    for(auto i = 0; i < grad_dir.num_elements(); ++i)
    {
        const auto direction = std::fabs(grad_dir[i]);
        grad_dir[i]          = (direction < 22.5 || direction >= 157.5) ? 0 : (direction < 67.5) ? 1 : (direction < 112.5) ? 2 : 3;
    }

    // Non-maximum suppression
    std::vector<int> strong_edges;
    const auto       upper_thresh = static_cast<uint32_t>(upper);
    const auto       lower_thresh = static_cast<uint32_t>(lower);

    const auto pixel_at_offset = [&](const SimpleTensor<unsigned_U> &tensor, const Coordinates & coord, int xoffset, int yoffset)
    {
        return tensor_elem_at(tensor, Coordinates{ coord.x() + xoffset, coord.y() + yoffset }, border_mode, static_cast<unsigned_U>(constant_border_value));
    };

    for(auto i = 0; i < dst.num_elements(); ++i)
    {
        const auto coord = index2coord(dst.shape(), i);
        if(!is_in_valid_region(valid_region, coord) || grad_mag[i] <= lower_thresh)
        {
            dst[i] = MARK_ZERO;
            continue;
        }

        unsigned_U mag_90;
        unsigned_U mag90;
        switch(grad_dir[i])
        {
            case 0: // North/South edge direction, compare against East/West pixels (left & right)
                mag_90 = pixel_at_offset(grad_mag, coord, -1, 0);
                mag90  = pixel_at_offset(grad_mag, coord, 1, 0);
                break;
            case 1: // NE/SW edge direction, compare against NW/SE pixels (top-left & bottom-right)
                mag_90 = pixel_at_offset(grad_mag, coord, -1, -1);
                mag90  = pixel_at_offset(grad_mag, coord, +1, +1);
                break;
            case 2: // East/West edge direction, compare against North/South pixels (top & bottom)
                mag_90 = pixel_at_offset(grad_mag, coord, 0, -1);
                mag90  = pixel_at_offset(grad_mag, coord, 0, +1);
                break;
            case 3: // NW/SE edge direction, compare against NE/SW pixels (top-right & bottom-left)
                mag_90 = pixel_at_offset(grad_mag, coord, +1, -1);
                mag90  = pixel_at_offset(grad_mag, coord, -1, +1);
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid gradient phase provided");
                break;
        }

        // Potential edge if greater than both pixels at +/-90° on either side
        if(grad_mag[i] > mag_90 && grad_mag[i] > mag90)
        {
            // Double thresholding and edge tracing
            if(grad_mag[i] > upper_thresh)
            {
                dst[i] = MARK_EDGE; // Definite edge pixel
                strong_edges.emplace_back(i);
            }
            else
            {
                dst[i] = MARK_MAYBE;
            }
        }
        else
        {
            dst[i] = MARK_ZERO; // Since not greater than neighbours
        }
    }

    // Final edge tracing
    trace_edge<T>(dst, valid_region);
    return dst;
}
} // namespace

template <typename T>
SimpleTensor<T> canny_edge_detector(const SimpleTensor<T> &src,
                                    int32_t upper_thresh, int32_t lower_thresh, int gradient_size, MagnitudeType norm_type,
                                    BorderMode border_mode, T constant_border_value)
{
    if(gradient_size < 7)
    {
        return canny_edge_detector_impl<int16_t>(src, upper_thresh, lower_thresh, gradient_size, norm_type, border_mode, constant_border_value);
    }
    else
    {
        return canny_edge_detector_impl<int32_t>(src, upper_thresh, lower_thresh, gradient_size, norm_type, border_mode, constant_border_value);
    }
}

template SimpleTensor<uint8_t> canny_edge_detector(const SimpleTensor<uint8_t> &src,
                                                   int32_t upper_thresh, int32_t lower_thresh, int gradient_size, MagnitudeType norm_type,
                                                   BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
