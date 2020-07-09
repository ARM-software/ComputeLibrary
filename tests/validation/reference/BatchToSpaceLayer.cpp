/*
 * Copyright (c) 2018 Arm Limited.
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
#include "BatchToSpaceLayer.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
// Batch to Space
template <typename T>
SimpleTensor<T> batch_to_space(const SimpleTensor<T> &src, const SimpleTensor<int32_t> &block_shape, const TensorShape &dst_shape)
{
    ARM_COMPUTE_ERROR_ON(block_shape[0] <= 0);
    ARM_COMPUTE_ERROR_ON(block_shape[1] <= 0);
    SimpleTensor<T> result(dst_shape, src.data_type());

    int        in_pos    = 0;
    const auto width_in  = static_cast<int>(src.shape()[0]);
    const auto height_in = static_cast<int>(src.shape()[1]);
    const auto z_in      = static_cast<int>(src.shape()[2]);
    const auto batch_in  = static_cast<int>(src.shape()[3]);

    for(int batch = 0; batch < batch_in; ++batch)
    {
        for(int z = 0; z < z_in; ++z)
        {
            for(int y = 0; y < height_in; ++y)
            {
                for(int x = 0; x < width_in; ++x)
                {
                    const int r       = src.shape()[3] / (block_shape[0] * block_shape[1]);
                    const int out_x   = (block_shape[0] * x + (batch / r) % block_shape[0]);
                    const int out_y   = (block_shape[1] * y + (batch / r) / block_shape[0]);
                    const int out_pos = out_x + dst_shape[0] * out_y + z * dst_shape[0] * dst_shape[1] + (batch % r) * dst_shape[0] * dst_shape[1] * dst_shape[2];
                    result[out_pos]   = src[in_pos];
                    ++in_pos;
                }
            }
        }
    }

    return result;
}
template SimpleTensor<float> batch_to_space(const SimpleTensor<float> &src, const SimpleTensor<int32_t> &block_shape, const TensorShape &dst_shape);
template SimpleTensor<half> batch_to_space(const SimpleTensor<half> &src, const SimpleTensor<int32_t> &block_shape, const TensorShape &dst_shape);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
