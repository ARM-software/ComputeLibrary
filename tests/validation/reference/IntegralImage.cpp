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
#include "IntegralImage.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<uint32_t> integral_image(const SimpleTensor<T> &src)
{
    SimpleTensor<uint32_t> dst(src.shape(), DataType::U32);

    // Length of dimensions
    const size_t width  = src.shape().x();
    const size_t height = src.shape().y();
    const size_t depth  = src.shape().total_size_upper(2);

    const size_t image_size = width * height;

    for(size_t z = 0; z < depth; ++z)
    {
        size_t current_image = z * image_size;

        //First element of each image
        dst[current_image] = src[current_image];

        // First row of each image (add only pixel on the left)
        for(size_t x = 1; x < width; ++x)
        {
            dst[current_image + x] = static_cast<uint32_t>(src[current_image + x]) + dst[current_image + x - 1];
        }

        // Subsequent rows
        for(size_t y = 1; y < height; ++y)
        {
            size_t current_row = current_image + (width * y);

            // First element of each row (add only pixel up)
            dst[current_row] = static_cast<uint32_t>(src[current_row]) + dst[current_row - width];

            // Following row elements
            for(size_t x = 1; x < width; ++x)
            {
                size_t current_pixel = current_row + x;

                // out = in + up(out) + left(out) - up_left(out)
                dst[current_pixel] = static_cast<uint32_t>(src[current_pixel]) + dst[current_pixel - 1]
                                     + dst[current_pixel - width] - dst[current_pixel - width - 1];
            }
        }
    }

    return dst;
}

template SimpleTensor<uint32_t> integral_image(const SimpleTensor<uint8_t> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
