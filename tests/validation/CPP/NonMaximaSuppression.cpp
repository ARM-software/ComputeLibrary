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
#include "NonMaximaSuppression.h"

#include "Utils.h"
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
SimpleTensor<T> non_maxima_suppression(const SimpleTensor<T> &src, BorderMode border_mode, T constant_border_value)
{
    constexpr int   block_size = 3;
    SimpleTensor<T> dst(src.shape(), src.data_type(), src.num_channels());
    ValidRegion     valid_region = shape_to_valid_region(src.shape(), border_mode == BorderMode::UNDEFINED, BorderSize(block_size / 2));

    for(int i = 0; i < src.num_elements(); ++i)
    {
        Coordinates coord = index2coord(src.shape(), i);
        int         x     = coord.x();
        int         y     = coord.y();

        if(!is_in_valid_region(valid_region, coord))
        {
            continue;
        }

        if(src[i] >= tensor_elem_at(src, Coordinates(x - 1, y - 1), border_mode, constant_border_value) && src[i] >= tensor_elem_at(src, Coordinates(x, y - 1), border_mode, constant_border_value)
           && src[i] >= tensor_elem_at(src, Coordinates(x + 1, y - 1), border_mode, constant_border_value) && src[i] >= tensor_elem_at(src, Coordinates(x - 1, y), border_mode, constant_border_value)
           && src[i] > tensor_elem_at(src, Coordinates(x + 1, y), border_mode, constant_border_value) && src[i] > tensor_elem_at(src, Coordinates(x - 1, y + 1), border_mode, constant_border_value)
           && src[i] > tensor_elem_at(src, Coordinates(x, y + 1), border_mode, constant_border_value) && src[i] > tensor_elem_at(src, Coordinates(x + 1, y + 1), border_mode, constant_border_value))
        {
            dst[i] = src[i];
        }
        else
        {
            dst[i] = T(0);
        }
    }

    return dst;
}

template SimpleTensor<float> non_maxima_suppression(const SimpleTensor<float> &src, BorderMode border_mode, float constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
