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
#include "ReorgLayer.h"

#include "arm_compute/core/Types.h"

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
TensorShape compute_reorg_shape(const TensorShape &src_shape, int32_t stride)
{
    ARM_COMPUTE_ERROR_ON(stride <= 0);

    TensorShape dst_shape = src_shape;
    dst_shape.set(0, src_shape.x() / stride);
    dst_shape.set(1, src_shape.y() / stride);
    dst_shape.set(2, src_shape.z() * stride * stride);

    return dst_shape;
}
} // namespace

template <typename T>
SimpleTensor<T> reorg_layer(const SimpleTensor<T> &src, int32_t stride)
{
    // Calculate output shape
    const TensorShape dst_shape = compute_reorg_shape(src.shape(), stride);

    // Create destination tensor
    SimpleTensor<T> dst{ dst_shape, src.data_type() };

    const unsigned int W          = dst.shape().x();
    const unsigned int H          = dst.shape().y();
    const unsigned int C          = dst.shape().z();
    const unsigned int out_c      = C / (stride * stride);
    const unsigned int outer_dims = dst.shape().total_size() / (W * H * C);

    // Calculate layer reorg in NCHW
    Coordinates map_coords;
    for(unsigned int b = 0; b < outer_dims; ++b)
    {
        map_coords.set(3, b);
        for(unsigned int c = 0; c < C; ++c)
        {
            map_coords.set(2, c % out_c);
            const unsigned int offset = c / out_c;
            for(unsigned int h = 0; h < H; ++h)
            {
                map_coords.set(1, h * stride + offset / stride);
                for(unsigned int w = 0; w < W; ++w)
                {
                    const unsigned int dst_idx = w + W * (h + H * (c + C * b));
                    map_coords.set(0, w * stride + offset % stride);
                    dst[dst_idx] = *reinterpret_cast<const T *>(src(map_coords));
                }
            }
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> reorg_layer(const SimpleTensor<uint8_t> &src, int32_t stride);
template SimpleTensor<uint16_t> reorg_layer(const SimpleTensor<uint16_t> &src, int32_t stride);
template SimpleTensor<uint32_t> reorg_layer(const SimpleTensor<uint32_t> &src, int32_t stride);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
