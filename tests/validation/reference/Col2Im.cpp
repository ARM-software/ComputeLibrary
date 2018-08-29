/*
 * Copyright (c) 2018 ARM Limited.
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
#include "Col2Im.h"

#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> col2im(const SimpleTensor<T> &src, const TensorShape &dst_shape, unsigned int num_groups)
{
    SimpleTensor<T> dst{ dst_shape, src.data_type(), 1 };

    // Compute reference
    const size_t batches    = dst_shape[3];
    const size_t src_width  = src.shape().x();
    const size_t src_height = src.shape().y();

    if(num_groups == 1)
    {
        // Batches are on the 3rd dimension of the input tensor
        int dst_idx = 0;
        for(size_t b = 0; b < batches; ++b)
        {
            for(size_t x = 0; x < src_width; ++x)
            {
                for(size_t y = 0; y < src_height; ++y)
                {
                    dst[dst_idx++] = src[coord2index(src.shape(), Coordinates(x, y, b))];
                }
            }
        }
    }
    else
    {
        int dst_idx = 0;
        for(size_t b = 0; b < batches; ++b)
        {
            for(size_t g = 0; g < num_groups; ++g)
            {
                for(size_t x = 0; x < src_width; ++x)
                {
                    for(size_t y = 0; y < src_height; ++y)
                    {
                        dst[dst_idx++] = src[coord2index(src.shape(), Coordinates(x, y, g, b))];
                    }
                }
            }
        }
    }
    return dst;
}

template SimpleTensor<float> col2im(const SimpleTensor<float> &src, const TensorShape &dst_shape, unsigned int num_groups);
template SimpleTensor<half> col2im(const SimpleTensor<half> &src, const TensorShape &dst_shape, unsigned int num_groups);
template SimpleTensor<uint8_t> col2im(const SimpleTensor<uint8_t> &src, const TensorShape &dst_shape, unsigned int num_groups);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
