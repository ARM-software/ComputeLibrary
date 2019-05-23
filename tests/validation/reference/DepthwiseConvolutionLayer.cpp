/*
 * Copyright (c) 2017-2019 ARM Limited.
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
SimpleTensor<T> depthwise_convolution(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &biases, const TensorShape &dst_shape, const PadStrideInfo &conv_info,
                                      unsigned int depth_multiplier, const Size2D &dilation, QuantizationInfo out_quant_info)
{
    ARM_COMPUTE_UNUSED(out_quant_info);

    SimpleTensor<T> dst{ dst_shape, src.data_type(), 1 };

    // Compute reference
    const int filter_width  = weights.shape().x();
    const int filter_height = weights.shape().y();
    const int filter_plane  = filter_width * filter_height;
    const int input_width   = src.shape().x();
    const int input_height  = src.shape().y();
    const int input_depth   = src.shape().z();
    const int num_batches   = src.shape().total_size() / (input_width * input_height * input_depth);

    const int pad_left   = conv_info.pad_left();
    const int pad_top    = conv_info.pad_top();
    const int pad_right  = conv_info.pad_right();
    const int pad_bottom = conv_info.pad_bottom();

    const float patch_width  = (filter_width + (dilation.x() - 1) * (filter_width - 1));
    const float patch_height = (filter_height + (dilation.y() - 1) * (filter_height - 1));

    const int patch_half_width_floor  = patch_width / 2;
    const int patch_half_height_floor = patch_height / 2;

    const auto patch_half_width_ceil  = static_cast<int>(std::ceil(patch_width / 2));
    const auto patch_half_height_ceil = static_cast<int>(std::ceil(patch_height / 2));

    const int minimum_x = -pad_left + patch_half_width_floor;
    const int minimum_y = -pad_top + patch_half_height_floor;
    const int maximum_x = input_width + pad_left + pad_right - static_cast<int>(patch_width);
    const int maximum_y = input_height + pad_top + pad_bottom - static_cast<int>(patch_height);

    const T border_value(0);

    int out_pos = 0;
    for(int r = 0; r < num_batches; ++r)
    {
        for(int z = 0; z < input_depth; ++z)
        {
            for(unsigned int m = 0; m < depth_multiplier; ++m)
            {
                const int out_z = z * depth_multiplier + m;

                for(int y = minimum_y; y <= minimum_y + maximum_y; y += conv_info.stride().second)
                {
                    for(int x = minimum_x; x <= minimum_x + maximum_x; x += conv_info.stride().first)
                    {
                        Coordinates coords(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(r));
                        size_t      filter_offset = filter_plane * out_z;

                        T val(0);
                        for(int j = y - patch_half_height_floor; j < y + patch_half_height_ceil; j += dilation.y())
                        {
                            for(int i = x - patch_half_width_floor; i < x + patch_half_width_ceil; i += dilation.x())
                            {
                                coords.set(0, i);
                                coords.set(1, j);
                                val += *(weights.data() + filter_offset) * tensor_elem_at(src, coords, BorderMode::CONSTANT, border_value);
                                ++filter_offset;
                            }
                        }

                        dst[out_pos++] = saturate_cast<T>(val + *static_cast<const TB *>(biases(Coordinates(out_z))));
                    }
                }
            }
        }
    }

    return dst;
}

template <>
SimpleTensor<uint8_t> depthwise_convolution(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &biases, const TensorShape &dst_shape,
                                            const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, QuantizationInfo out_quant_info)
{
    // if no explicit quantization has been set you the same as src
    if(out_quant_info == QuantizationInfo(0.0f, 0))
    {
        out_quant_info = src.quantization_info();
    }
    SimpleTensor<uint8_t> dst{ dst_shape, src.data_type(), 1, out_quant_info };

    // Create reference
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

    const int pad_left   = conv_info.pad_left();
    const int pad_top    = conv_info.pad_top();
    const int pad_right  = conv_info.pad_right();
    const int pad_bottom = conv_info.pad_bottom();

    const float patch_width  = (filter_width + (dilation.x() - 1) * (filter_width - 1));
    const float patch_height = (filter_height + (dilation.y() - 1) * (filter_height - 1));

    const int patch_half_width_floor  = patch_width / 2;
    const int patch_half_height_floor = patch_height / 2;

    const auto patch_half_width_ceil  = static_cast<int>(std::ceil(patch_width / 2));
    const auto patch_half_height_ceil = static_cast<int>(std::ceil(patch_height / 2));

    const int minimum_x = -pad_left + patch_half_width_floor;
    const int minimum_y = -pad_top + patch_half_height_floor;
    const int maximum_x = input_width + pad_left + pad_right - static_cast<int>(patch_width);
    const int maximum_y = input_height + pad_top + pad_bottom - static_cast<int>(patch_height);

    int out_pos = 0;
    for(int r = 0; r < num_batches; ++r)
    {
        for(int z = 0; z < input_depth; ++z)
        {
            for(unsigned int m = 0; m < depth_multiplier; ++m)
            {
                const int     out_z    = z * depth_multiplier + m;
                const int32_t bias_val = *static_cast<const int32_t *>(biases(Coordinates(out_z)));

                for(int y = minimum_y; y <= minimum_y + maximum_y; y += conv_info.stride().second)
                {
                    for(int x = minimum_x; x <= minimum_x + maximum_x; x += conv_info.stride().first)
                    {
                        Coordinates coords(x, y, z, r);
                        int         filter_offset = filter_plane * out_z;

                        int32_t val = 0;
                        for(int j = y - patch_half_height_floor; j < y + patch_half_height_ceil; j += dilation.y())
                        {
                            for(int i = x - patch_half_width_floor; i < x + patch_half_width_ceil; i += dilation.x())
                            {
                                coords.set(0, i);
                                coords.set(1, j);
                                const auto    in_val = tensor_elem_at<uint8_t>(src, coords, BorderMode::CONSTANT, -input_offset);
                                const uint8_t w_val  = *(weights.data() + filter_offset);
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
    }

    return dst;
}

template SimpleTensor<float> depthwise_convolution(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &biases, const TensorShape &dst_shape,
                                                   const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, QuantizationInfo out_quant_info);

template SimpleTensor<half> depthwise_convolution(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &biases, const TensorShape &dst_shape,
                                                  const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, QuantizationInfo out_quant_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
