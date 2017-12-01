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

#include "Gaussian5x5.h"
#include "Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> gaussian5x5(const SimpleTensor<T> &src, BorderMode border_mode, T constant_border_value)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());
    const std::array<T, 25> filter{ {
            1, 4, 6, 4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1, 4, 6, 4, 1
        } };
    const float scale = 1.f / 256.f;
    for(int element_idx = 0; element_idx < src.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(src.shape(), element_idx);
        apply_2d_spatial_filter(id, src, dst, TensorShape(5U, 5U), filter.data(), scale, border_mode, constant_border_value);
    }
    return dst;
}

template SimpleTensor<uint8_t> gaussian5x5(const SimpleTensor<uint8_t> &src, BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
