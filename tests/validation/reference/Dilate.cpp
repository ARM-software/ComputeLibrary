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
#include "Dilate.h"

#include "Utils.h"
#include "tests/validation/Helpers.h"

#include <algorithm>
#include <array>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> dilate(const SimpleTensor<T> &src, BorderMode border_mode, T constant_border_value)
{
    /*
             -1   x  +1
         -1 [tl][tc][tr] -1
          y [ml][xy][mr]  y
         +1 [bl][bc][br] +1
             -1   x  +1
        dilate:
        dst(x, y) = max[ src(x', y') for x-1<=x'<=x+1, y-1<=y'<=y+1 ] = max({tl, tc, tr, ml, xy, mr, bl, bc, br})
    */
    SimpleTensor<T> dst(src.shape(), src.data_type());

    for(int i = 0; i < src.num_elements(); ++i)
    {
        Coordinates coord = index2coord(src.shape(), i);
        const int   x     = coord.x();
        const int   y     = coord.y();

        std::array<T, 9> neighbours = { { 0 } };
        for(int row = y - 1, j = 0; row <= y + 1; ++row)
        {
            for(int col = x - 1; col <= x + 1; ++col, ++j)
            {
                coord.set(0, col);
                coord.set(1, row);
                neighbours[j] = tensor_elem_at(src, coord, border_mode, constant_border_value);
            }
        }

        dst[i] = *std::max_element(neighbours.cbegin(), neighbours.cend());
    }

    return dst;
}

template SimpleTensor<uint8_t> dilate(const SimpleTensor<uint8_t> &src, BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
