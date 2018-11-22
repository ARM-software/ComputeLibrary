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

    auto width_out  = static_cast<int>(dst_shape[0]);
    auto height_out = static_cast<int>(dst_shape[1]);
    auto z_out      = static_cast<int>(dst_shape[2]);

    int out_pos = 0;
    for(int batch = 0; batch < static_cast<int>(dst_shape[3]); ++batch)
    {
        for(int z = 0; z < z_out; ++z)
        {
            for(int y = 0; y < height_out; ++y)
            {
                for(int x = 0; x < width_out; ++x)
                {
                    if(x < paddings[0] || x > width_out - paddings[1] - 1
                       || y < paddings[2] || y > height_out - paddings[3] - 1)
                    {
                        result[out_pos] = 0;
                    }
                    else
                    {
                        const int r      = dst_shape[3] / (block_shape[0] * block_shape[1]);
                        const int in_x   = (block_shape[0] * (x - paddings[0]) + (batch / r) % block_shape[0]);
                        const int in_y   = (block_shape[1] * (y - paddings[2]) + (batch / r) / block_shape[0]);
                        int       in_pos = in_x + src.shape()[0] * in_y + z * src.shape()[0] * src.shape()[1] + (batch % r) * src.shape()[0] * src.shape()[1] * src.shape()[2];
                        result[out_pos]  = src[in_pos];
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
