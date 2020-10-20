/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "UpsampleLayer.h"

#include "support/Requires.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> upsample_layer(const SimpleTensor<T> &src, const Size2D &info, const InterpolationPolicy policy)
{
    ARM_COMPUTE_ERROR_ON(policy != InterpolationPolicy::NEAREST_NEIGHBOR);
    ARM_COMPUTE_UNUSED(policy);

    TensorShape output_shape = src.shape();
    output_shape.set(0, src.shape().x() * info.x());
    output_shape.set(1, src.shape().y() * info.y());

    // Create reference
    const int       stride_x   = info.x();
    const int       stride_y   = info.y();
    int             width_out  = output_shape.x();
    int             height_out = output_shape.y();
    SimpleTensor<T> out{ output_shape, src.data_type(), 1, src.quantization_info() };

    const int width_in      = src.shape().x();
    const int height_in     = src.shape().y();
    const int num_2d_slices = src.shape().total_size() / (width_in * height_in);

    for(int slice = 0; slice < num_2d_slices; ++slice)
    {
        const int offset_slice_in  = slice * width_in * height_in;
        const int offset_slice_out = slice * height_out * width_out;
        for(int y = 0; y < height_out; ++y)
        {
            for(int x = 0; x < width_out; ++x)
            {
                const int out_offset = y * width_out + x;
                const int in_offset  = (y / stride_y) * width_in + x / stride_x;

                T       *_out = out.data() + offset_slice_out + out_offset;
                const T *in   = src.data() + offset_slice_in + in_offset;
                *_out         = *in;
            }
        }
    }
    return out;
}

template SimpleTensor<float> upsample_layer(const SimpleTensor<float> &src,
                                            const Size2D &info, const InterpolationPolicy policy);
template SimpleTensor<half> upsample_layer(const SimpleTensor<half> &src,
                                           const Size2D &info, const InterpolationPolicy policy);
template SimpleTensor<uint8_t> upsample_layer(const SimpleTensor<uint8_t> &src,
                                              const Size2D &info, const InterpolationPolicy policy);
template SimpleTensor<int8_t> upsample_layer(const SimpleTensor<int8_t> &src,
                                             const Size2D &info, const InterpolationPolicy policy);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
