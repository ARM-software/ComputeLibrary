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
#include "Utils.h"

#include "tests/validation_new/Helpers.h"
#include "tests/validation_new/half.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
// Return a tensor element at a specified coordinate with different border modes
template <typename T>
T tensor_elem_at(const SimpleTensor<T> &in, Coordinates coord, BorderMode border_mode, T constant_border_value)
{
    const int  x      = coord.x();
    const int  y      = coord.y();
    const auto width  = static_cast<int>(in.shape().x());
    const auto height = static_cast<int>(in.shape().y());

    // If coordinates beyond range of tensor's width or height
    if(x < 0 || y < 0 || x >= width || y >= height)
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
    return in[coord2index(in.shape(), coord)];
}
template float tensor_elem_at(const SimpleTensor<float> &in, Coordinates coord, BorderMode border_mode, float constant_border_value);
template uint8_t tensor_elem_at(const SimpleTensor<uint8_t> &in, Coordinates coord, BorderMode border_mode, uint8_t constant_border_value);

// Return the bilinear value at a specified coordinate with different border modes
template <typename T>
T bilinear_policy(const SimpleTensor<T> &in, Coordinates id, float xn, float yn, BorderMode border_mode, uint8_t constant_border_value)
{
    int idx = std::floor(xn);
    int idy = std::floor(yn);

    const float dx   = xn - idx;
    const float dy   = yn - idy;
    const float dx_1 = 1.0f - dx;
    const float dy_1 = 1.0f - dy;

    id.set(0, idx);
    id.set(1, idy);
    const T tl = tensor_elem_at(in, id, border_mode, constant_border_value);
    id.set(0, idx + 1);
    id.set(1, idy);
    const T tr = tensor_elem_at(in, id, border_mode, constant_border_value);
    id.set(0, idx);
    id.set(1, idy + 1);
    const T bl = tensor_elem_at(in, id, border_mode, constant_border_value);
    id.set(0, idx + 1);
    id.set(1, idy + 1);
    const T br = tensor_elem_at(in, id, border_mode, constant_border_value);

    return tl * (dx_1 * dy_1) + tr * (dx * dy_1) + bl * (dx_1 * dy) + br * (dx * dy);
}
template uint8_t bilinear_policy(const SimpleTensor<uint8_t> &in, Coordinates id, float xn, float yn, BorderMode border_mode, uint8_t constant_border_value);

} // namespace validation
} // namespace test
} // namespace arm_compute
