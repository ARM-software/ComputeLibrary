/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_VALIDATION_UTILS_H
#define ARM_COMPUTE_TEST_VALIDATION_UTILS_H

#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Types.h"

#include <array>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Checks if a pixel has valid coordinates
 *
 * @param x            X coordinate
 * @param y            Y coordinate
 * @param width        Width of the image
 * @param height       Height of the image
 * @param border_size  Border size
 *
 * @return True if pixel is valid else false
 */
inline bool is_valid_pixel_index(int x, int y, int width, int height, int border_size)
{
    return ((x >= -border_size) && (y >= -border_size) && (x < (width + border_size)) && (y < height + border_size));
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"

// Return a tensor element at a specified coordinate with different border modes
template <typename T>
T tensor_elem_at(const SimpleTensor<T> &src, Coordinates coord, BorderMode border_mode, T constant_border_value)
{
    const int x      = coord.x();
    const int y      = coord.y();
    const int z      = coord.z();
    const int width  = src.shape().x();
    const int height = src.shape().y();
    const int depth  = src.shape().z();

    // If coordinates beyond range of tensor's width or height
    if(x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth)
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

    return src[coord2index(src.shape(), coord)];
}

#pragma GCC diagnostic pop

template <typename T>
T bilinear_policy(const SimpleTensor<T> &in, Coordinates id, float xn, float yn, BorderMode border_mode, T constant_border_value);

/* Apply 2D spatial filter on a single element of @p in at coordinates @p coord
 *
 * - filter sizes have to be odd number
 * - Row major order of filter assumed
 * - TO_ZERO rounding policy assumed
 * - SATURATE convert policy assumed
 */
template <typename T, typename U, typename V>
void apply_2d_spatial_filter(Coordinates coord, const SimpleTensor<T> &src, SimpleTensor<U> &dst, const TensorShape &filter_shape, const V *filter_itr, double scale, BorderMode border_mode,
                             T constant_border_value = T(0))
{
    double    val = 0.;
    const int x   = coord.x();
    const int y   = coord.y();
    for(int j = y - static_cast<int>(filter_shape[1] / 2); j <= y + static_cast<int>(filter_shape[1] / 2); ++j)
    {
        for(int i = x - static_cast<int>(filter_shape[0] / 2); i <= x + static_cast<int>(filter_shape[0] / 2); ++i)
        {
            coord.set(0, i);
            coord.set(1, j);
            val += static_cast<double>(*filter_itr) * tensor_elem_at(src, coord, border_mode, constant_border_value);
            ++filter_itr;
        }
    }
    coord.set(0, x);
    coord.set(1, y);
    dst[coord2index(src.shape(), coord)] = saturate_cast<U>(support::cpp11::trunc(val * scale));
}

RawTensor transpose(const RawTensor &src, int chunk_width = 1);

bool valid_bilinear_policy(float xn, float yn, int width, int height, BorderMode border_mode);
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_VALIDATION_UTILS_H */
