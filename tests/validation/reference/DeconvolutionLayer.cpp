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
#include "ConvolutionLayer.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename TB>
SimpleTensor<T> deconvolution_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, const TensorShape &output_shape,
                                    const PadStrideInfo &info)
{
    // Create reference
    const int stride_x           = info.stride().first;
    const int stride_y           = info.stride().second;
    const int weights_width      = weights.shape().x();
    const int weights_height     = weights.shape().y();
    const int weights_upper_dims = weights.shape().total_size() / (weights_width * weights_height);

    // Find the upsampled dimensions
    unsigned int out_x = (src.shape().x() - 1) * stride_x + 1;
    unsigned int out_y = (src.shape().y() - 1) * stride_y + 1;

    // Find the padding needed for the convolution with stride 1 in order to match output shape
    unsigned int padx = output_shape.x() - (out_x - weights_width + 1);
    unsigned int pady = output_shape.y() - (out_y - weights_height + 1);
    out_x += padx;
    out_y += pady;

    TensorShape scaled_shape = src.shape();
    scaled_shape.set(0, out_x);
    scaled_shape.set(1, out_y);
    SimpleTensor<T> scaled{ scaled_shape, src.data_type(), 1, src.quantization_info() };

    const int width_in      = src.shape().x();
    const int height_in     = src.shape().y();
    const int width_scaled  = scaled.shape().x();
    const int height_scaled = scaled.shape().y();
    const int num_2d_slices = src.shape().total_size() / (width_in * height_in);
    ARM_COMPUTE_ERROR_ON(info.pad().first > (weights.shape().x() - 1));

    if(src.data_type() == DataType::QASYMM8)
    {
        const uint8_t quantized_zero = src.quantization_info().uniform().offset;
        std::fill_n(scaled.data(), scaled.num_elements(), quantized_zero);
    }
    else
    {
        std::fill_n(scaled.data(), scaled.num_elements(), T(0));
    }

    // Flip weights by 180 degrees
    SimpleTensor<T> weights_flipped{ weights.shape(), weights.data_type(), 1, weights.quantization_info() };
    for(int ud = 0; ud < weights_upper_dims; ++ud)
    {
        const int offset = ud * weights_width * weights_height;
        for(int y = 0; y < weights_height; ++y)
        {
            for(int x = 0; x < weights_width; ++x)
            {
                weights_flipped[offset + (weights_height - 1 - y) * weights_width + (weights_width - 1 - x)] = weights[offset + y * weights_width + x];
            }
        }
    }

    for(int slice = 0; slice < num_2d_slices; ++slice)
    {
        const int offset_slice_in  = slice * width_in * height_in;
        const int offset_slice_out = slice * width_scaled * height_scaled;
        const int start_x          = padx / 2;
        const int start_y          = pady / 2;
        const int end_y            = height_scaled - pady / 2;
        const int end_x            = width_scaled - padx / 2;

        for(int yi = start_y, in_y = 0; yi < end_y; yi += stride_y, in_y++)
        {
            for(int xi = start_x, in_x = 0; xi < end_x; xi += stride_x, in_x++)
            {
                const T *in  = src.data() + offset_slice_in + in_y * width_in + in_x;
                T       *out = scaled.data() + offset_slice_out + xi + yi * width_scaled;
                *out         = *in;
            }
        }
    }

    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
    return convolution_layer(scaled, weights_flipped, bias, output_shape, conv_info);
}

template SimpleTensor<uint8_t> deconvolution_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &bias, const TensorShape &output_shape,
                                                   const PadStrideInfo &info);
template SimpleTensor<float> deconvolution_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &output_shape,
                                                 const PadStrideInfo &info);
template SimpleTensor<half> deconvolution_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &bias, const TensorShape &output_shape,
                                                const PadStrideInfo &info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
