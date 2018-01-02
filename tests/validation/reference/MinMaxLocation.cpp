/*
 * Copyright (c) 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal src the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included src all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. src NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER src AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR src CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "MinMaxLocation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
void compute_min_max(const SimpleTensor<T> &src, T &min, T &max)
{
    // Set min and max to first pixel
    min = src[0];
    max = src[0];

    ARM_COMPUTE_ERROR_ON(src.num_elements() == 0);

    // Look for min and max values
    for(int i = 1; i < src.num_elements(); ++i)
    {
        if(src[i] < min)
        {
            min = src[i];
        }
        if(src[i] > max)
        {
            max = src[i];
        }
    }
}

template <typename T>
MinMaxLocationValues<T> min_max_location(const SimpleTensor<T> &src)
{
    MinMaxLocationValues<T> dst;

    const size_t width = src.shape().x();

    compute_min_max<T>(src, dst.min, dst.max);

    Coordinates2D coord{ 0, 0 };

    for(int i = 0; i < src.num_elements(); ++i)
    {
        coord.x = static_cast<int32_t>(i % width);
        coord.y = static_cast<int32_t>(i / width);

        if(src[i] == dst.min)
        {
            dst.min_loc.push_back(coord);
        }
        if(src[i] == dst.max)
        {
            dst.max_loc.push_back(coord);
        }
    }

    return dst;
}

template MinMaxLocationValues<uint8_t> min_max_location(const SimpleTensor<uint8_t> &src);
template MinMaxLocationValues<int16_t> min_max_location(const SimpleTensor<int16_t> &src);
template MinMaxLocationValues<float> min_max_location(const SimpleTensor<float> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
