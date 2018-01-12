/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "DepthwiseConvolutionLayer.h"

#include "ConvolutionLayer.h"
#include "Utils.h"

#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Utils.h"
#include "tests/validation/reference/UtilsQuantizedAsymm.h"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

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
template <typename T, typename TB>
SimpleTensor<T> depthwise_convolution(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &biases, const TensorShape &dst_shape, const PadStrideInfo &conv_info)
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

                    T val(0);
                    for(int j = y - filter_half_height; j <= static_cast<int>(y + filter_half_height); ++j)
                    {
                        for(int i = x - filter_half_width; i <= static_cast<int>(x + filter_half_width); ++i)
                        {
                            coords.set(0, i);
                            coords.set(1, j);
                            T border_value(0);
                            val += *(weights.data() + filter_offset) * tensor_elem_at(src, coords, BorderMode::CONSTANT, border_value);
                            ++filter_offset;
                        }
                    }
                    coords.set(0, x);
                    coords.set(1, y);
                    dst[out_pos++] = saturate_cast<T>(val + *static_cast<const TB *>(biases(Coordinates(z))));
                }
            }
        }
    }

    return dst;
}

template <>
SimpleTensor<uint8_t> depthwise_convolution(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &biases, const TensorShape &dst_shape,
                                            const PadStrideInfo &conv_info)
{
    // Create reference
    SimpleTensor<uint8_t> dst{ dst_shape, src.data_type(), 1, src.fixed_point_position(), src.quantization_info() };

    const int   input_offset   = -src.quantization_info().offset;
    const float input_scale    = src.quantization_info().scale;
    const int   weights_offset = -weights.quantization_info().offset;
    const float weights_scale  = weights.quantization_info().scale;
    const int   output_offset  = dst.quantization_info().offset;
    const float output_scale   = dst.quantization_info().scale;

    int         output_multiplier;
    int         output_shift;
    const float multiplier = input_scale * weights_scale / output_scale;
    arm_compute::quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

    // Compute reference
    const int filter_width  = weights.shape().x();
    const int filter_height = weights.shape().y();
    const int filter_plane  = filter_width * filter_height;
    const int input_width   = src.shape().x();
    const int input_height  = src.shape().y();
    const int input_depth   = src.shape().z();
    const int num_batches   = src.shape().total_size() / (input_width * input_height * input_depth);

    const int filter_half_size = filter_width / 2;
    const int pad_x            = std::min(filter_half_size, static_cast<int>(conv_info.pad().first));
    const int pad_y            = std::min(filter_half_size, static_cast<int>(conv_info.pad().second));
    const int minimum_x        = -pad_x + filter_half_size;
    const int minimum_y        = -pad_y + filter_half_size;

    int out_pos = 0;
    for(int r = 0; r < num_batches; ++r)
    {
        for(int z = 0; z < input_depth; ++z)
        {
            int32_t bias_val = *static_cast<const int32_t *>(biases(Coordinates(z)));
            for(int y = minimum_y; y < input_height + pad_y - filter_half_size; y += conv_info.stride().second)
            {
                for(int x = minimum_x; x < input_width + pad_x - filter_half_size; x += conv_info.stride().first)
                {
                    Coordinates coords(x, y, z, r);
                    int         filter_offset = filter_plane * z;

                    int32_t val = 0;
                    for(int j = y - filter_half_size; j <= (y + filter_half_size); ++j)
                    {
                        for(int i = x - filter_half_size; i <= (x + filter_half_size); ++i)
                        {
                            coords.set(0, i);
                            coords.set(1, j);
                            auto    in_val = tensor_elem_at<uint8_t>(src, coords, BorderMode::CONSTANT, -input_offset);
                            uint8_t w_val  = *(weights.data() + filter_offset);
                            val += (in_val + input_offset) * (w_val + weights_offset);
                            ++filter_offset;
                        }
                    }
                    val += bias_val;
                    val = asymm_rounding_divide_by_pow2(asymm_int_mult(val, output_multiplier), output_shift);
                    val += output_offset;
                    val = std::max<int32_t>(val, 0);
                    val = std::min<int32_t>(val, 255);

                    // Store the result
                    dst[out_pos++] = val;
                }
            }
        }
    }

    return dst;
}

template SimpleTensor<float> depthwise_convolution(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &biases, const TensorShape &dst_shape,
                                                   const PadStrideInfo &conv_info);

template SimpleTensor<half> depthwise_convolution(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &biases, const TensorShape &dst_shape,
                                                  const PadStrideInfo &conv_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
