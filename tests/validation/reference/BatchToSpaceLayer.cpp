/*
 * Copyright (c) 2018, 2023 Arm Limited.
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
SimpleTensor<T> batch_to_space(const SimpleTensor<T> &src, const SimpleTensor<int32_t> &block_shape, const TensorShape &dst_shape, const CropInfo &crop_info)
{
    ARM_COMPUTE_ERROR_ON(block_shape[0] <= 0);
    ARM_COMPUTE_ERROR_ON(block_shape[1] <= 0);
    SimpleTensor<T> result(dst_shape, src.data_type());
    int             out_pos    = 0;
    const auto      width_out  = static_cast<int>(dst_shape[0]);
    const auto      height_out = static_cast<int>(dst_shape[1]);
    const auto      z_out      = static_cast<int>(dst_shape[2]);
    const auto      batch_out  = static_cast<int>(dst_shape[3]);
    ARM_COMPUTE_ERROR_ON(width_out <= static_cast<int>(crop_info.left + crop_info.right));
    ARM_COMPUTE_ERROR_ON(height_out <= static_cast<int>(crop_info.top + crop_info.bottom));

    for(int batch = 0; batch < batch_out; ++batch)
    {
        for(int z = 0; z < z_out; ++z)
        {
            for(int y = 0; y < height_out; ++y)
            {
                for(int x = 0; x < width_out; ++x)
                {
                    const int x_c      = x + crop_info.left;
                    const int y_c      = y + crop_info.top;
                    const int in_batch = batch + ((x_c % block_shape[0]) + (y_c % block_shape[1]) * (block_shape[0])) * dst_shape[3];
                    const int in_x     = x_c / block_shape[0];
                    const int in_y     = y_c / block_shape[1];
                    const int in_pos   = in_x + src.shape()[0] * in_y + z * src.shape()[0] * src.shape()[1] + in_batch * src.shape()[0] * src.shape()[1] * src.shape()[2];
                    result[out_pos]    = src[in_pos];
                    ++out_pos;
                }
            }
        }
    }

    return result;
}
template SimpleTensor<float> batch_to_space(const SimpleTensor<float> &src, const SimpleTensor<int32_t> &block_shape, const TensorShape &dst_shape, const CropInfo &crop_info = CropInfo{});
template SimpleTensor<half> batch_to_space(const SimpleTensor<half> &src, const SimpleTensor<int32_t> &block_shape, const TensorShape &dst_shape, const CropInfo &crop_info = CropInfo{});
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
