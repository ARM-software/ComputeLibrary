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
#include "Transpose.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> transpose(const SimpleTensor<T> &src)
{
    // Make rows the columns of the original shape
    TensorShape dst_shape{ src.shape().y(), src.shape().x() };

    // Create reference
    SimpleTensor<T> dst{ dst_shape, src.data_type() };

    // Compute reference
    for(int i = 0; i < src.num_elements(); ++i)
    {
        const Coordinates coord = index2coord(src.shape(), i);
        const Coordinates dst_coord{ coord.y(), coord.x() };
        const size_t      dst_index = coord2index(dst.shape(), dst_coord);

        dst[dst_index] = src[i];
    }

    return dst;
}

template SimpleTensor<uint8_t> transpose(const SimpleTensor<uint8_t> &src);
template SimpleTensor<uint16_t> transpose(const SimpleTensor<uint16_t> &src);
template SimpleTensor<uint32_t> transpose(const SimpleTensor<uint32_t> &src);
template SimpleTensor<half> transpose(const SimpleTensor<half> &src);
template SimpleTensor<float> transpose(const SimpleTensor<float> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
