/*
 * Copyright (c) 2017-2020 Arm Limited.
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
namespace
{
/** Perform a depthwise convolution for floating-point types
 *
 * - Three dimensions tensors
 * - Third dimention is number of channels
 * - Depths of input tensor and filter are equals
 * - Padding, stride and output shape "match"
 *
 */
template <typename T>
SimpleTensor<T> depthwise_convolution_fp(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<T> &biases, const TensorShape &dst_shape, const PadStrideInfo &conv_info,
                                         unsigned int depth_multiplier, const Size2D &dilation, const QuantizationInfo &out_quant_info)
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

    const int pad_left = conv_info.pad_left();
    const int pad_top  = conv_info.pad_top();

    const float patch_width  = (filter_width + (dilation.x() - 1) * (filter_width - 1));
    const float patch_height = (filter_height + (dilation.y() - 1) * (filter_height - 1));

    const int patch_half_width_floor  = patch_width / 2;
    const int patch_half_height_floor = patch_height / 2;

    const auto patch_half_width_ceil  = static_cast<int>(std::ceil(patch_width / 2));
    const auto patch_half_height_ceil = static_cast<int>(std::ceil(patch_height / 2));

    const int minimum_x = -pad_left + patch_half_width_floor;
    const int minimum_y = -pad_top + patch_half_height_floor;
    const int maximum_x = (conv_info.stride().first * (dst_shape[0] - 1));
    const int maximum_y = (conv_info.stride().second * (dst_shape[1] - 1));

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

                        dst[out_pos++] = saturate_cast<T>(val + *static_cast<const T *>(biases(Coordinates(out_z))));
                    }
                }
            }
        }
    }

    return dst;
}

/** Perform a quantized depthwise convolution
 *
 * - Three dimensions tensors
 * - Third dimention is number of channels
 * - Depths of input tensor and filter are equals
 * - Padding, stride and output shape "match"
 * - QASYMM8/QASYMM8_SIGNED input, output
 * - QASYMM8/QASYMM8_SIGNED or QSYMM8_PER_CHANNEL filter
 *
 */
template <typename T, typename TW, typename TB>
SimpleTensor<T> depthwise_convolution_quantized(const SimpleTensor<T> &src, const SimpleTensor<TW> &weights, const SimpleTensor<int32_t> &biases, const TensorShape &dst_shape,
                                                const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, const QuantizationInfo &out_quant_info)
{
    // if no explicit quantization has been set you the same as src
    const QuantizationInfo &dst_qinfo = out_quant_info.uniform().empty() ? src.quantization_info() : out_quant_info;
    SimpleTensor<T>         dst{ dst_shape, src.data_type(), 1, dst_qinfo };

    // Create reference
    const int   input_offset   = -src.quantization_info().uniform().offset;
    const float input_scale    = src.quantization_info().uniform().scale;
    const int   weights_offset = -weights.quantization_info().uniform().offset;
    const int   output_offset  = dst_qinfo.uniform().offset;
    const float output_scale   = dst_qinfo.uniform().scale;

    const std::vector<float> weights_scale_vec = weights.quantization_info().scale();

    // Compute reference
    const int filter_width  = weights.shape().x();
    const int filter_height = weights.shape().y();
    const int filter_plane  = filter_width * filter_height;
    const int input_width   = src.shape().x();
    const int input_height  = src.shape().y();
    const int input_depth   = src.shape().z();
    const int num_batches   = src.shape().total_size() / (input_width * input_height * input_depth);

    const int pad_left = conv_info.pad_left();
    const int pad_top  = conv_info.pad_top();

    const float patch_width  = (filter_width + (dilation.x() - 1) * (filter_width - 1));
    const float patch_height = (filter_height + (dilation.y() - 1) * (filter_height - 1));

    const int patch_half_width_floor  = patch_width / 2;
    const int patch_half_height_floor = patch_height / 2;

    const auto patch_half_width_ceil  = static_cast<int>(std::ceil(patch_width / 2));
    const auto patch_half_height_ceil = static_cast<int>(std::ceil(patch_height / 2));

    const int minimum_x = -pad_left + patch_half_width_floor;
    const int minimum_y = -pad_top + patch_half_height_floor;
    const int maximum_x = (conv_info.stride().first * (dst_shape[0] - 1));
    const int maximum_y = (conv_info.stride().second * (dst_shape[1] - 1));

    const bool is_quantized_per_channel = is_data_type_quantized_per_channel(weights.data_type());

    const int min = std::numeric_limits<T>::lowest();
    const int max = std::numeric_limits<T>::max();

    int out_pos = 0;
    for(int r = 0; r < num_batches; ++r)
    {
        for(int z = 0; z < input_depth; ++z)
        {
            for(unsigned int m = 0; m < depth_multiplier; ++m)
            {
                const int     out_z    = z * depth_multiplier + m;
                const int32_t bias_val = *static_cast<const int32_t *>(biases(Coordinates(out_z)));

                int         output_multiplier = 0;
                int         output_shift      = 0;
                const float weights_scale     = (is_quantized_per_channel) ? weights_scale_vec[out_z] : weights_scale_vec[0];
                const float multiplier        = input_scale * weights_scale / output_scale;
                arm_compute::quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);

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
                                const auto in_val = tensor_elem_at<T>(src, coords, BorderMode::CONSTANT, -input_offset);
                                const TW   w_val  = *(weights.data() + filter_offset);
                                val += (in_val + input_offset) * (w_val + weights_offset);
                                ++filter_offset;
                            }
                        }
                        val += bias_val;
                        // Quantize down
                        val = quantize_down_scale_by_fixedpoint(val, output_multiplier, output_shift, output_offset, min, max);

                        // Store the result
                        dst[out_pos++] = val;
                    }
                }
            }
        }
    }

    return dst;
}
} // namespace

template <>
SimpleTensor<float> depthwise_convolution(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &biases, const TensorShape &dst_shape,
                                          const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, const QuantizationInfo &out_quant_info)
{
    return depthwise_convolution_fp(src, weights, biases, dst_shape, conv_info, depth_multiplier, dilation, out_quant_info);
}

template <>
SimpleTensor<half> depthwise_convolution(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &biases, const TensorShape &dst_shape,
                                         const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, const QuantizationInfo &out_quant_info)
{
    return depthwise_convolution_fp(src, weights, biases, dst_shape, conv_info, depth_multiplier, dilation, out_quant_info);
}

template <>
SimpleTensor<uint8_t> depthwise_convolution(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &biases, const TensorShape &dst_shape,
                                            const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, const QuantizationInfo &out_quant_info)
{
    return depthwise_convolution_quantized<uint8_t, uint8_t, int32_t>(src, weights, biases, dst_shape, conv_info, depth_multiplier, dilation, out_quant_info);
}

template <>
SimpleTensor<uint8_t> depthwise_convolution(const SimpleTensor<uint8_t> &src, const SimpleTensor<int8_t> &weights, const SimpleTensor<int32_t> &biases, const TensorShape &dst_shape,
                                            const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, const QuantizationInfo &out_quant_info)
{
    return depthwise_convolution_quantized<uint8_t, int8_t, int32_t>(src, weights, biases, dst_shape, conv_info, depth_multiplier, dilation, out_quant_info);
}

template <>
SimpleTensor<int8_t> depthwise_convolution(const SimpleTensor<int8_t> &src, const SimpleTensor<int8_t> &weights, const SimpleTensor<int32_t> &biases, const TensorShape &dst_shape,
                                           const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation, const QuantizationInfo &out_quant_info)
{
    return depthwise_convolution_quantized<int8_t, int8_t, int32_t>(src, weights, biases, dst_shape, conv_info, depth_multiplier, dilation, out_quant_info);
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
