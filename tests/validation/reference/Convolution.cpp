/*
 * Copyright (c) 2017, 2018 ARM Limited.
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

#include "Convolution.h"
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
SimpleTensor<T> convolution(const SimpleTensor<T> &src, const int16_t *conv, uint32_t scale, BorderMode border_mode, T constant_border_value, const unsigned int width, const unsigned int height)
{
    SimpleTensor<T>       dst(src.shape(), src.data_type());
    SimpleTensor<int32_t> sum(src.shape(), src.data_type());

    for(int element_idx = 0; element_idx < src.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(src.shape(), element_idx);
        apply_2d_spatial_filter(id, src, sum, TensorShape(width, height), conv, 1, border_mode, constant_border_value);

        if(tensor_elem_at<int32_t>(sum, id, border_mode, constant_border_value) < 0)
        {
            dst[element_idx] = 0;
        }
        else if((tensor_elem_at<int32_t>(sum, id, border_mode, constant_border_value) / scale) > 255)
        {
            dst[element_idx] = 255;
        }
        else
        {
            dst[element_idx] = tensor_elem_at<int32_t>(sum, id, border_mode, constant_border_value) / scale;
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> convolution(const SimpleTensor<uint8_t> &src, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value,
                                           const unsigned int widht, const unsigned int height);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
