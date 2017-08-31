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
#include "DepthwiseConvolution.h"

#include "ConvolutionLayer.h"
#include "Utils.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
/** Perform a depthwise convolution
 *
 * - Three dimensions tensors
 * - Third dimention is number of channels
 * - Depths of input tensor and filter are equals
 * - Padding, stride and output shape "match"
 *
 */
template <typename T>
SimpleTensor<T> depthwise_convolution(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const TensorShape &dst_shape, const PadStrideInfo &conv_info)
{
    // Create reference
    SimpleTensor<T> dst{ dst_shape, src.data_type(), 1, src.fixed_point_position() };

    // Compute reference
    const size_t filter_width  = weights.shape().x();
    const size_t filter_height = weights.shape().y();
    const size_t filter_plane  = filter_width * filter_height;
    const size_t input_width   = src.shape().x();
    const size_t input_height  = src.shape().y();
    const size_t input_depth   = src.shape().z();

    const size_t filter_half_size = filter_width / 2;
    const size_t pad_x            = std::min(filter_half_size, static_cast<size_t>(conv_info.pad().first));
    const size_t pad_y            = std::min(filter_half_size, static_cast<size_t>(conv_info.pad().second));
    const size_t minimum_x        = -pad_x + filter_half_size;
    const size_t minimum_y        = -pad_y + filter_half_size;

    int out_pos = 0;
    for(size_t z = 0; z < input_depth; ++z)
    {
        for(size_t y = minimum_y; y < input_height + pad_y - filter_half_size; y += conv_info.stride().second)
        {
            for(size_t x = minimum_x; x < input_width + pad_x - filter_half_size; x += conv_info.stride().first)
            {
                Coordinates coords(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z));
                size_t      filter_offset = filter_plane * z;

                T val = 0;
                for(int j = y - filter_half_size; j <= static_cast<int>(y + filter_half_size); ++j)
                {
                    for(int i = x - filter_half_size; i <= static_cast<int>(x + filter_half_size); ++i)
                    {
                        coords.set(0, i);
                        coords.set(1, j);
                        val += *(weights.data() + filter_offset) * tensor_elem_at(src, coords, BorderMode::CONSTANT, 0.f);
                        ++filter_offset;
                    }
                }
                coords.set(0, x);
                coords.set(1, y);
                dst[out_pos++] = saturate_cast<T>(val);
            }
        }
    }

    return dst;
}

template SimpleTensor<float> depthwise_convolution(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const TensorShape &dst_shape, const PadStrideInfo &conv_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
