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
#include "SpaceToBatch.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
// Space to Batch
template <typename T>
SimpleTensor<T> space_to_batch(const SimpleTensor<T> &src, const SimpleTensor<int32_t> &block_shape, const SimpleTensor<int32_t> &paddings, const TensorShape &dst_shape)
{
    SimpleTensor<T> result(dst_shape, src.data_type());

    const auto width_out  = static_cast<int>(dst_shape[0]);
    const auto height_out = static_cast<int>(dst_shape[1]);
    const auto batch_out  = static_cast<int>(dst_shape[3]);

    const auto width_in  = static_cast<int>(src.shape()[0]);
    const auto height_in = static_cast<int>(src.shape()[1]);
    const auto batch_in  = static_cast<int>(src.shape()[3]);

    const auto channel = static_cast<int>(src.shape()[2]);

    const auto block_width  = block_shape[0];
    const auto block_height = block_shape[1];

    const auto padding_left = paddings[0];
    const auto padding_top  = paddings[2];

    int out_pos = 0;
    for(int outB = 0; outB < batch_out; ++outB)
    {
        unsigned int inB = outB % batch_in;

        int shift_w = (outB / batch_in) % block_width;
        int shift_h = (outB / batch_in) / block_width;

        for(int c = 0; c < channel; ++c)
        {
            for(int outH = 0; outH < height_out; ++outH)
            {
                for(int outW = 0; outW < width_out; ++outW)
                {
                    const auto in_pos = ((inB * channel + c) * height_in + ((outH * block_height + shift_h) - padding_top)) * width_in + (outW * block_width + shift_w) - padding_left;

                    if(outH * block_height + shift_h < padding_top || outH * block_height + shift_h >= padding_top + height_in || outW * block_width + shift_w < padding_left
                       || outW * block_width + shift_w >= padding_left + width_in)
                    {
                        result[out_pos] = 0;
                    }
                    else
                    {
                        result[out_pos] = src[in_pos];
                    }
                    ++out_pos;
                }
            }
        }
    }
    return result;
}

template SimpleTensor<float> space_to_batch(const SimpleTensor<float> &src, const SimpleTensor<int32_t> &block_shape, const SimpleTensor<int32_t> &paddings, const TensorShape &dst_shape);
template SimpleTensor<half> space_to_batch(const SimpleTensor<half> &src, const SimpleTensor<int32_t> &block_shape, const SimpleTensor<int32_t> &paddings, const TensorShape &dst_shape);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
