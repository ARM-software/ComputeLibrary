/*
 * Copyright (c) 2019 ARM Limited.
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
#include "SpaceToDepth.h"

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
SimpleTensor<T> space_to_depth(const SimpleTensor<T> &src, const TensorShape &dst_shape, const int block_shape)
{
    SimpleTensor<T> result(dst_shape, src.data_type());

    const auto width_out   = static_cast<int>(dst_shape[0]);
    const auto height_out  = static_cast<int>(dst_shape[1]);
    const auto channel_out = static_cast<int>(dst_shape[2]);

    const auto width_in   = static_cast<int>(src.shape()[0]);
    const auto height_in  = static_cast<int>(src.shape()[1]);
    const auto channel_in = static_cast<int>(src.shape()[2]);

    const auto batch = static_cast<int>(src.shape()[3]);

    const auto block_width  = block_shape;
    const auto block_height = block_shape;

    int out_pos = 0;
    for(int ba = 0; ba < batch; ++ba)
    {
        for(int outC = 0; outC < channel_out; ++outC)
        {
            unsigned int inC = outC % channel_in;

            int shift_w = (outC / channel_in) % block_width;
            int shift_h = (outC / channel_in) / block_width;

            for(int outH = 0; outH < height_out; ++outH)
            {
                for(int outW = 0; outW < width_out; ++outW)
                {
                    const auto in_pos = ((ba * channel_in + inC) * height_in + ((outH * block_height + shift_h))) * width_in + (outW * block_width + shift_w);
                    result[out_pos]   = src[in_pos];
                    ++out_pos;
                }
            }
        }
    }
    return result;
}

template SimpleTensor<float> space_to_depth(const SimpleTensor<float> &src, const TensorShape &dst_shape, const int block_shape);
template SimpleTensor<half> space_to_depth(const SimpleTensor<half> &src, const TensorShape &dst_shape, const int block_shape);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
