/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "FastCorners.h"

#include "Utils.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/NonMaximaSuppression.h"

#include "tests/framework/Asserts.h"
#include <iomanip>

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
constexpr unsigned int bresenham_radius = 3;
constexpr unsigned int bresenham_count  = 16;

/*
    Offsets of the 16 pixels in the Bresenham circle of radius 3 centered on P
        . . . . . . . . .
        . . . F 0 1 . . .
        . . E . . . 2 . .
        . D . . . . . 3 .
        . C . . P . . 4 .
        . B . . . . . 5 .
        . . A . . . 6 . .
        . . . 9 8 7 . . .
        . . . . . . . . .
*/
const std::array<std::array<int, 2>, 16> circle_offsets =
{
    {
        { { 0, -3 } },  // 0 - pixel #1
        { { 1, -3 } },  // 1 - pixel #2
        { { 2, -2 } },  // 2 - pixel #3
        { { 3, -1 } },  // 3 - pixel #4
        { { 3, 0 } },   // 4 - pixel #5
        { { 3, 1 } },   // 5 - pixel #6
        { { 2, 2 } },   // 6 - pixel #7
        { { 1, 3 } },   // 7 - pixel #8
        { { 0, 3 } },   // 8 - pixel #9
        { { -1, 3 } },  // 9 - pixel #10
        { { -2, 2 } },  // A - pixel #11
        { { -3, 1 } },  // B - pixel #12
        { { -3, 0 } },  // C - pixel #13
        { { -3, -1 } }, // D - pixel #14
        { { -2, -2 } }, // E - pixel #15
        { { -1, -3 } }  // F - pixel #16
    }
};

/*
    FAST-9 bit masks for consecutive points surrounding a corner candidate
    Rejection of non-corners is expedited by checking pixels 1, 9, then 5, 13...
*/
const std::array<uint16_t, 16> fast9_masks =
{
    {
        0x01FF, // 0000 0001 1111 1111
        0x03FE, // 0000 0011 1111 1110
        0x07FC, // 0000 0111 1111 1100
        0x0FF8, // 0000 1111 1111 1000
        0x1FF0, // 0001 1111 1111 0000
        0x3FE0, // 0011 1111 1110 0000
        0x7FC0, // 0111 1111 1100 0000
        0xFF80, // 1111 1111 1000 0000
        0xFF01, // 1111 1111 0000 0001
        0xFE03, // 1111 1110 0000 0011
        0xFC07, // 1111 1100 0000 0111
        0xF80F, // 1111 1000 0000 1111
        0xF01F, // 1111 0000 0001 1111
        0xE03F, // 1110 0000 0011 1111
        0xC07F, // 1100 0000 0111 1111
        0x80FF  // 1000 0000 1111 1111
    }
};

inline bool in_range(const uint8_t low, const uint8_t high, const uint8_t val)
{
    return low <= val && val <= high;
}

template <typename T, typename F>
bool is_a_corner(const Coordinates &candidate, const SimpleTensor<T> &src, uint8_t threshold, BorderMode border_mode, T constant_border_value, F intensity_at)
{
    const auto intensity_p   = tensor_elem_at(src, candidate, border_mode, constant_border_value);
    const auto thresh_bright = intensity_p + threshold;
    const auto thresh_dark   = intensity_p - threshold;

    // Quicker rejection of non-corner points by checking pixels 1, 9 then 5, 13 around the candidate
    const auto p1  = intensity_at(candidate, 0);
    const auto p9  = intensity_at(candidate, 8);
    const auto p5  = intensity_at(candidate, 4);
    const auto p13 = intensity_at(candidate, 12);

    if((in_range(thresh_dark, thresh_bright, p1) && in_range(thresh_dark, thresh_bright, p9))
       || (in_range(thresh_dark, thresh_bright, p5) && in_range(thresh_dark, thresh_bright, p13)))
    {
        return false;
    }

    uint16_t mask_bright = 0;
    uint16_t mask_dark   = 0;

    // Set bits of the brighter/darker pixels mask accordingly
    for(unsigned int n = 0; n < bresenham_count; ++n)
    {
        T intensity_n = intensity_at(candidate, n);
        mask_bright |= (intensity_n > thresh_bright) << n;
        mask_dark |= (intensity_n < thresh_dark) << n;
    }

    // Mark as corner candidate if brighter/darker pixel sequence satisfies any one of the FAST-9 masks
    const auto found = std::find_if(fast9_masks.begin(), fast9_masks.end(), [&](decltype(fast9_masks[0]) mask)
    {
        return (mask_bright & mask) == mask || (mask_dark & mask) == mask;
    });

    return found != fast9_masks.end();
}
} // namespace

template <typename T>
std::vector<KeyPoint> fast_corners(const SimpleTensor<T> &src, float input_thresh, bool suppress_nonmax, BorderMode border_mode, T constant_border_value)
{
    // Get intensity of pixel at given index on the Bresenham circle around a candidate point
    const auto intensity_at = [&](const Coordinates & point, const unsigned int idx)
    {
        const auto  offset = circle_offsets[idx];
        Coordinates px{ point.x() + offset[0], point.y() + offset[1] };
        return tensor_elem_at(src, px, border_mode, constant_border_value);
    };

    const auto            threshold = static_cast<uint8_t>(input_thresh);
    std::vector<KeyPoint> corners;

    // 1. Detect potential corners (the segment test)
    std::vector<Coordinates> corner_candidates;
    SimpleTensor<uint8_t>    scores(src.shape(), DataType::U8);
    ValidRegion              valid_region = shape_to_valid_region(src.shape(), BorderMode::UNDEFINED == border_mode, BorderSize(bresenham_radius));

    for(int i = 0; i < src.num_elements(); ++i)
    {
        Coordinates candidate = index2coord(src.shape(), i);
        scores[i]             = 0;
        if(!is_in_valid_region(valid_region, candidate))
        {
            continue;
        }

        if(is_a_corner(candidate, src, threshold, border_mode, constant_border_value, intensity_at))
        {
            corner_candidates.emplace_back(candidate);
            scores[i] = 1;
        }
    }

    // 2. Calculate corner scores if necessary
    if(suppress_nonmax)
    {
        for(const auto &candidate : corner_candidates)
        {
            const auto index      = coord2index(scores.shape(), candidate);
            uint8_t    thresh_max = UINT8_MAX;
            uint8_t    thresh_min = threshold;
            uint8_t    response   = (thresh_min + thresh_max) / 2;

            // Corner score (response) is the largest threshold for which the pixel remains a corner
            while(thresh_max - thresh_min > 1)
            {
                response = (thresh_min + thresh_max) / 2;
                if(is_a_corner(candidate, src, response, border_mode, constant_border_value, intensity_at))
                {
                    thresh_min = response; // raise threshold
                }
                else
                {
                    thresh_max = response; // lower threshold
                }
            }
            scores[index] = thresh_min;
        }

        scores       = non_maxima_suppression(scores, border_mode, constant_border_value);
        valid_region = shape_to_valid_region(scores.shape(), BorderMode::UNDEFINED == border_mode, BorderSize(bresenham_radius + 1));
    }

    for(const auto &candidate : corner_candidates)
    {
        const auto index = coord2index(scores.shape(), candidate);
        if(scores[index] > 0.f && is_in_valid_region(valid_region, candidate))
        {
            KeyPoint corner;
            corner.x               = candidate.x();
            corner.y               = candidate.y();
            corner.strength        = scores[index];
            corner.tracking_status = 1;
            corner.scale           = 0.f;
            corner.orientation     = 0.f;
            corner.error           = 0.f;
            corners.emplace_back(corner);
        }
    }

    return corners;
}

template std::vector<KeyPoint> fast_corners(const SimpleTensor<uint8_t> &src, float threshold, bool suppress_nonmax, BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
