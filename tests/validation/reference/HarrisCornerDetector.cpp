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
#include "HarrisCornerDetector.h"

#include "Utils.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/NonMaximaSuppression.h"
#include "tests/validation/reference/Sobel.h"

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
template <typename T>
std::tuple<SimpleTensor<T>, SimpleTensor<T>, float> compute_sobel(const SimpleTensor<uint8_t> &src, int gradient_size, int block_size, BorderMode border_mode, uint8_t constant_border_value)
{
    SimpleTensor<T> grad_x;
    SimpleTensor<T> grad_y;
    float           norm_factor = 0.f;

    std::tie(grad_x, grad_y) = sobel<T>(src, gradient_size, border_mode, constant_border_value, GradientDimension::GRAD_XY);

    switch(gradient_size)
    {
        case 3:
            norm_factor = 1.f / (4 * 255 * block_size);
            break;
        case 5:
            norm_factor = 1.f / (16 * 255 * block_size);
            break;
        case 7:
            norm_factor = 1.f / (64 * 255 * block_size);
            break;
        default:
            ARM_COMPUTE_ERROR("Gradient size not supported.");
    }

    return std::make_tuple(grad_x, grad_y, norm_factor);
}

template <typename T, typename U>
std::vector<KeyPoint> harris_corner_detector_impl(const SimpleTensor<U> &src, float threshold, float min_dist, float sensitivity, int gradient_size, int block_size, BorderMode border_mode,
                                                  U constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(block_size != 3 && block_size != 5 && block_size != 7);

    SimpleTensor<T> grad_x;
    SimpleTensor<T> grad_y;
    float           norm_factor = 0.f;

    // Sobel
    std::tie(grad_x, grad_y, norm_factor) = compute_sobel<T>(src, gradient_size, block_size, border_mode, constant_border_value);

    SimpleTensor<float> scores(src.shape(), DataType::F32);
    ValidRegion         scores_region = shape_to_valid_region(scores.shape(), border_mode == BorderMode::UNDEFINED, BorderSize(gradient_size / 2 + block_size / 2));

    // Calculate scores
    for(int i = 0; i < scores.num_elements(); ++i)
    {
        Coordinates src_coord = index2coord(src.shape(), i);
        Coordinates block_top_left{ src_coord.x() - block_size / 2, src_coord.y() - block_size / 2 };
        Coordinates block_bottom_right{ src_coord.x() + block_size / 2, src_coord.y() + block_size / 2 };

        if(!is_in_valid_region(scores_region, src_coord))
        {
            scores[i] = 0.f;
            continue;
        }

        float Gx2 = 0.f;
        float Gy2 = 0.f;
        float Gxy = 0.f;

        // Calculate Gx^2, Gy^2 and Gxy within the given window
        for(int y = block_top_left.y(); y <= block_bottom_right.y(); ++y)
        {
            for(int x = block_top_left.x(); x <= block_bottom_right.x(); ++x)
            {
                Coordinates block_coord(x, y);

                const float norm_x = tensor_elem_at(grad_x, block_coord, border_mode, static_cast<T>(constant_border_value)) * norm_factor;
                const float norm_y = tensor_elem_at(grad_y, block_coord, border_mode, static_cast<T>(constant_border_value)) * norm_factor;

                Gx2 += std::pow(norm_x, 2);
                Gy2 += std::pow(norm_y, 2);
                Gxy += norm_x * norm_y;
            }
        }

        const float trace2   = std::pow(Gx2 + Gy2, 2);
        const float det      = Gx2 * Gy2 - std::pow(Gxy, 2);
        const float response = det - sensitivity * trace2;

        if(response > threshold)
        {
            scores[i] = response;
        }
        else
        {
            scores[i] = 0.f;
        }
    }

    // Suppress non-maxima candidates
    SimpleTensor<float> suppressed_scores        = non_maxima_suppression(scores, border_mode != BorderMode::UNDEFINED ? BorderMode::CONSTANT : BorderMode::UNDEFINED, 0.f);
    ValidRegion         suppressed_scores_region = shape_to_valid_region(suppressed_scores.shape(), border_mode == BorderMode::UNDEFINED, BorderSize(gradient_size / 2 + block_size / 2 + 1));

    // Create vector of candidate corners
    std::vector<KeyPoint> corner_candidates;

    for(int i = 0; i < suppressed_scores.num_elements(); ++i)
    {
        Coordinates coord = index2coord(suppressed_scores.shape(), i);

        if(is_in_valid_region(suppressed_scores_region, coord) && suppressed_scores[i] != 0.f)
        {
            KeyPoint corner;
            corner.x               = coord.x();
            corner.y               = coord.y();
            corner.tracking_status = 1;
            corner.strength        = suppressed_scores[i];
            corner.scale           = 0.f;
            corner.orientation     = 0.f;
            corner.error           = 0.f;

            corner_candidates.emplace_back(corner);
        }
    }

    // Sort descending by strength
    std::sort(corner_candidates.begin(), corner_candidates.end(), [](const KeyPoint & a, const KeyPoint & b)
    {
        return a.strength > b.strength;
    });

    std::vector<KeyPoint> corners;
    corners.reserve(corner_candidates.size());

    // Only add corner if there is no stronger within min_dist
    for(const KeyPoint &point : corner_candidates)
    {
        const auto strongest = std::find_if(corners.begin(), corners.end(), [&](const KeyPoint & other)
        {
            return std::sqrt((std::pow(point.x - other.x, 2) + std::pow(point.y - other.y, 2))) < min_dist;
        });

        if(strongest == corners.end())
        {
            corners.emplace_back(point);
        }
    }

    corners.shrink_to_fit();

    return corners;
}
} // namespace

template <typename T>
std::vector<KeyPoint> harris_corner_detector(const SimpleTensor<T> &src, float threshold, float min_dist, float sensitivity, int gradient_size, int block_size, BorderMode border_mode,
                                             T constant_border_value)
{
    if(gradient_size < 7)
    {
        return harris_corner_detector_impl<int16_t>(src, threshold, min_dist, sensitivity, gradient_size, block_size, border_mode, constant_border_value);
    }
    else
    {
        return harris_corner_detector_impl<int32_t>(src, threshold, min_dist, sensitivity, gradient_size, block_size, border_mode, constant_border_value);
    }
}

template std::vector<KeyPoint> harris_corner_detector(const SimpleTensor<uint8_t> &src, float threshold, float min_dist, float sensitivity, int gradient_size, int block_size, BorderMode border_mode,
                                                      uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
