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
    const int filter_width  = weights.shape().x();
    const int filter_height = weights.shape().y();
    const int filter_plane  = filter_width * filter_height;
    const int input_width   = src.shape().x();
    const int input_height  = src.shape().y();
    const int input_depth   = src.shape().z();
    const int num_batches   = src.shape().total_size() / (input_width * input_height * input_depth);

    const int filter_half_width  = filter_width / 2;
    const int filter_half_height = filter_height / 2;

    const int pad_left   = std::min(static_cast<int>(conv_info.pad_left()), filter_half_width);
    const int pad_top    = std::min(static_cast<int>(conv_info.pad_top()), filter_half_height);
    const int pad_right  = std::min(static_cast<int>(conv_info.pad_right()), filter_half_width);
    const int pad_bottom = std::min(static_cast<int>(conv_info.pad_bottom()), filter_half_height);

    const int minimum_x = -pad_left + filter_half_width;
    const int minimum_y = -pad_top + filter_half_height;
    const int maximum_x = input_width + pad_left - filter_half_width + pad_right - filter_half_width;
    const int maximum_y = input_height + pad_top - filter_half_height + pad_bottom - filter_half_height;

    int out_pos = 0;
    for(int r = 0; r < num_batches; ++r)
    {
        for(int z = 0; z < input_depth; ++z)
        {
            for(int y = minimum_y; y < minimum_y + maximum_y; y += conv_info.stride().second)
            {
                for(int x = minimum_x; x < minimum_x + maximum_x; x += conv_info.stride().first)
                {
                    Coordinates coords(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(r));
                    size_t      filter_offset = filter_plane * z;

                    T val = 0;
                    for(int j = y - filter_half_height; j <= static_cast<int>(y + filter_half_height); ++j)
                    {
                        for(int i = x - filter_half_width; i <= static_cast<int>(x + filter_half_width); ++i)
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
    }

    return dst;
}

template SimpleTensor<float> depthwise_convolution(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const TensorShape &dst_shape, const PadStrideInfo &conv_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
