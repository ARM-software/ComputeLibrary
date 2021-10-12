/*
 * Copyright (c) 2021 Arm Limited.
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
#include "Conv3D.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

// Source/Destination Tensor shape indices (N D H W C)
constexpr unsigned int batch_dim   = 4u;
constexpr unsigned int depth_dim   = 3u;
constexpr unsigned int height_dim  = 2u;
constexpr unsigned int width_dim   = 1u;
constexpr unsigned int channel_dim = 0u;

// Weight tensor shape indices (D H W Cin Cout)
constexpr unsigned int weights_depth_dim  = 4u;
constexpr unsigned int weights_height_dim = 3u;
constexpr unsigned int weights_width_dim  = 2u;
constexpr unsigned int weights_CHin_dim   = 1u;
constexpr unsigned int weights_CHout_dim  = 0u;

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
inline bool is_valid_pixel(int i, int min, int max)
{
    return (i >= min && i < max);
}
// Evaluate the weights against an element in a given tensor.
template <typename T>
T calculate_conv3d(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const Size3D &dilation, int batch,
                   int z_start, int y_start, int x_start, int ch_out)
{
    const unsigned int weights_width  = weights.shape()[weights_width_dim];
    const unsigned int weights_height = weights.shape()[weights_height_dim];
    const unsigned int weights_depth  = weights.shape()[weights_depth_dim];

    const unsigned int src_channels = src.shape()[channel_dim];
    const unsigned int src_width    = src.shape()[width_dim];
    const unsigned int src_height   = src.shape()[height_dim];
    const unsigned int src_depth    = src.shape()[depth_dim];

    T total(0);
    for(unsigned int weight_d = 0; weight_d < weights_depth; ++weight_d)
    {
        const int idx_z = z_start + dilation.depth * weight_d;
        for(unsigned int weight_y = 0; weight_y < weights_height; ++weight_y)
        {
            const int idx_y = y_start + dilation.height * weight_y;
            for(unsigned int weight_x = 0; weight_x < weights_width; ++weight_x)
            {
                const int idx_x = x_start + dilation.width * weight_x;

                //Check if the point is within padding
                const bool is_x_valid       = is_valid_pixel(idx_x, 0, src_width);
                const bool is_y_valid       = is_valid_pixel(idx_y, 0, src_height);
                const bool is_z_valid       = is_valid_pixel(idx_z, 0, src_depth);
                const bool is_invalid_pixel = !(is_x_valid && is_y_valid && is_z_valid);
                if(is_invalid_pixel)
                {
                    continue;
                }

                for(unsigned int ch_in = 0; ch_in < src_channels; ++ch_in)
                {
                    const T *in_ptr = src.data();
                    const T *w_ptr  = weights.data();

                    const int in_offset     = coord2index(src.shape(), Coordinates{ ch_in, idx_x, idx_y, idx_z, batch });
                    const int weight_offset = coord2index(weights.shape(), Coordinates{ ch_out, ch_in, weight_x, weight_y, weight_d });
                    T         input_value   = in_ptr[in_offset];
                    T         weight_value  = w_ptr[weight_offset];
                    total += (input_value * weight_value);
                }
            }
        }
    }
    return total;
}
}

template <typename T>
SimpleTensor<T> conv3d(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<T> &bias, SimpleTensor<T> &dst, const Conv3dInfo &conv3d_info)
{
    // Compute reference
    const unsigned int batch_size     = src.shape()[batch_dim];
    const unsigned int dst_width      = dst.shape()[width_dim];
    const unsigned int dst_height     = dst.shape()[height_dim];
    const unsigned int dst_depth      = dst.shape()[depth_dim];
    const unsigned int src_channels   = src.shape()[channel_dim];
    const unsigned int weights_out_ch = weights.shape()[weights_CHout_dim];
    const unsigned int dst_channels   = dst.shape()[channel_dim];
    const size_t       pad_left       = conv3d_info.padding.left;
    const size_t       pad_top        = conv3d_info.padding.top;
    const size_t       pad_front      = conv3d_info.padding.front;
    const size_t       stride_x       = conv3d_info.stride.x();
    const size_t       stride_y       = conv3d_info.stride.y();
    const size_t       stride_z       = conv3d_info.stride.z();

    const TensorShape dst_shape = arm_compute::misc::shape_calculator::compute_conv3d_shape(src.shape(), weights.shape(), conv3d_info);

    ARM_COMPUTE_UNUSED(src_channels, weights_out_ch, dst_channels, dst_shape, weights_CHin_dim);
    // Number of batches of source and destination tensors must match.
    ARM_COMPUTE_ERROR_ON(src.shape()[batch_dim] != dst.shape()[batch_dim]);
    // Input channels in the source and weights must match.
    ARM_COMPUTE_ERROR_ON(src_channels != weights.shape()[weights_CHin_dim]);
    // Weight channels in the destination and weights must match.
    ARM_COMPUTE_ERROR_ON(weights_out_ch != dst_channels);
    // Bias must match the number of destination channels.
    ARM_COMPUTE_ERROR_ON(bias.shape()[0] != dst_channels);
    // Compare given dst tensor shape with expected shape.
    ARM_COMPUTE_ERROR_ON(dst.shape() != dst_shape);

    for(unsigned int batch = 0; batch < batch_size; ++batch)
    {
        for(unsigned int z_out = 0; z_out < dst_depth; ++z_out)
        {
            const int z_start = (z_out * stride_z) - pad_front;
            for(unsigned int y_out = 0; y_out < dst_height; ++y_out)
            {
                const int y_start = (y_out * stride_y) - pad_top;
                for(unsigned int x_out = 0; x_out < dst_width; ++x_out)
                {
                    const int x_start = (x_out * stride_x) - pad_left;
                    for(unsigned int ch_out = 0; ch_out < dst_channels; ++ch_out)
                    {
                        T weighted_value = calculate_conv3d<T>(src, weights, conv3d_info.dilation, batch, z_start,
                                                               y_start, x_start, ch_out);
                        T        *out_ptr = dst.data();
                        const T *b_ptr   = bias.data();
                        T         bias_value(0);
                        const int out_offset = coord2index(dst.shape(), Coordinates{ ch_out, x_out, y_out, z_out, batch });
                        bias_value           = b_ptr[ch_out];
                        out_ptr[out_offset]  = weighted_value + bias_value;
                    }
                }
            }
        }
    }
    return dst;
}

template SimpleTensor<float> conv3d(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, SimpleTensor<float> &dst,
                                    const Conv3dInfo &conv3d_info);
template SimpleTensor<half> conv3d(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &bias, SimpleTensor<half> &dst,
                                   const Conv3dInfo &conv3d_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute