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
#include "arm_compute/core/Helpers.h"

#include "Median3x3.h"
#include "Utils.h"

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
constexpr unsigned int filter_size = 3;              /* Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size(filter_size / 2); /* Border size of the kernel/filter around its central element. */
} // namespace

template <typename T>
SimpleTensor<T> median3x3(const SimpleTensor<T> &src, BorderMode border_mode, T constant_border_value)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());
    const int       size_tot_filter = filter_size * filter_size;

    for(int src_idx = 0; src_idx < src.num_elements(); ++src_idx)
    {
        std::array<T, size_tot_filter> filter_elems = { { 0 } };
        Coordinates id = index2coord(src.shape(), src_idx);
        const int   x  = id.x();
        const int   y  = id.y();

        for(int j = y - static_cast<int>(border_size.top), index = 0; j <= y + static_cast<int>(border_size.bottom); ++j)
        {
            for(int i = x - static_cast<int>(border_size.left); i <= x + static_cast<int>(border_size.right); ++i, ++index)
            {
                id.set(0, i);
                id.set(1, j);
                filter_elems[index] = tensor_elem_at(src, id, border_mode, constant_border_value);
            }
        }
        std::sort(filter_elems.begin(), filter_elems.end());
        dst[src_idx] = filter_elems[size_tot_filter / 2];
    }

    return dst;
}

template SimpleTensor<uint8_t> median3x3(const SimpleTensor<uint8_t> &src, BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
