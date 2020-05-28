/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> reorg_layer(const SimpleTensor<T> &src, int32_t stride)
{
    ARM_COMPUTE_ERROR_ON(src.shape().num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(src.data_layout() != DataLayout::NCHW);

    TensorInfo        input_info(src.shape(), 1, src.data_type());
    const TensorShape output_shape = misc::shape_calculator::compute_reorg_output_shape(input_info, stride);

    // Create destination tensor
    SimpleTensor<T> dst{ output_shape, src.data_type() };

    const unsigned int W          = dst.shape().x();
    const unsigned int H          = dst.shape().y();
    const unsigned int C          = dst.shape().z();
    const unsigned int out_c      = C / (stride * stride);
    const unsigned int outer_dims = dst.shape().total_size() / (W * H * C);

    // Calculate layer reorg in NCHW
    Coordinates map_coords;

#if defined(_OPENMP)
    #pragma omp parallel for private(map_coords)
#endif /* _OPENMP */
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

template SimpleTensor<int32_t> reorg_layer(const SimpleTensor<int32_t> &src, int32_t stride);
template SimpleTensor<int16_t> reorg_layer(const SimpleTensor<int16_t> &src, int32_t stride);
template SimpleTensor<int8_t> reorg_layer(const SimpleTensor<int8_t> &src, int32_t stride);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
