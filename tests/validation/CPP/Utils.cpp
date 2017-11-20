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

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
// Return the bilinear value at a specified coordinate with different border modes
template <typename T>
T bilinear_policy(const SimpleTensor<T> &in, Coordinates id, float xn, float yn, BorderMode border_mode, T constant_border_value)
{
    const int idx = std::floor(xn);
    const int idy = std::floor(yn);

    const float dx   = xn - idx;
    const float dy   = yn - idy;
    const float dx_1 = 1.0f - dx;
    const float dy_1 = 1.0f - dy;

    const T border_value = constant_border_value;

    id.set(0, idx);
    id.set(1, idy);
    const float tl = tensor_elem_at(in, id, border_mode, border_value);
    id.set(0, idx + 1);
    id.set(1, idy);
    const float tr = tensor_elem_at(in, id, border_mode, border_value);
    id.set(0, idx);
    id.set(1, idy + 1);
    const float bl = tensor_elem_at(in, id, border_mode, border_value);
    id.set(0, idx + 1);
    id.set(1, idy + 1);
    const float br = tensor_elem_at(in, id, border_mode, border_value);

    return static_cast<T>(tl * (dx_1 * dy_1) + tr * (dx * dy_1) + bl * (dx_1 * dy) + br * (dx * dy));
}

template uint8_t bilinear_policy(const SimpleTensor<uint8_t> &in, Coordinates id, float xn, float yn, BorderMode border_mode, uint8_t constant_border_value);
template int16_t bilinear_policy(const SimpleTensor<int16_t> &in, Coordinates id, float xn, float yn, BorderMode border_mode, int16_t constant_border_value);
template half bilinear_policy(const SimpleTensor<half> &in, Coordinates id, float xn, float yn, BorderMode border_mode, half constant_border_value);
template float bilinear_policy(const SimpleTensor<float> &in, Coordinates id, float xn, float yn, BorderMode border_mode, float constant_border_value);

RawTensor transpose(const RawTensor &src, int chunk_width)
{
    // Create reference
    TensorShape dst_shape(src.shape());
    dst_shape.set(0, src.shape().y() * chunk_width);
    dst_shape.set(1, std::ceil(src.shape().x() / static_cast<float>(chunk_width)));

    RawTensor dst{ dst_shape, src.data_type() };

    // Compute reference
    uint8_t *out_ptr = dst.data();

    for(int i = 0; i < dst.num_elements(); i += chunk_width)
    {
        Coordinates coord   = index2coord(dst.shape(), i);
        size_t      coord_x = coord.x();
        coord.set(0, coord.y() * chunk_width);
        coord.set(1, coord_x / chunk_width);

        const int num_elements = std::min<int>(chunk_width, src.shape().x() - coord.x());

        std::copy_n(static_cast<const uint8_t *>(src(coord)), num_elements * src.element_size(), out_ptr);

        out_ptr += chunk_width * dst.element_size();
    }

    return dst;
}

bool valid_bilinear_policy(float xn, float yn, int width, int height, BorderMode border_mode)
{
    if(border_mode != BorderMode::UNDEFINED)
    {
        return true;
    }
    if((0 <= yn + 1) && (yn + 1 < height) && (0 <= xn + 1) && (xn + 1 < width))
    {
        return true;
    }
    return false;
}
} // namespace validation
} // namespace test
} // namespace arm_compute
