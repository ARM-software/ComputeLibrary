/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_DIRECTCONV2D_IMPL_H
#define ACL_SRC_CPU_KERNELS_DIRECTCONV2D_IMPL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/kernels/detail/NEDirectConvolutionDetail.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <algorithm>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
template <typename T, bool has_pads>
void linearize_volume_nchw(const uint8_t *const in_ptr,
                           T                   *out_ptr,
                           bool                 has_bias,
                           int                  top_left_x,
                           int                  top_left_y,
                           int                  kernel_width,
                           int                  kernel_height,
                           int                  kernel_depth,
                           int                  input_w,
                           int                  input_h,
                           int                  input_stride_x,
                           int                  input_stride_y,
                           int                  input_stride_z,
                           int                  pad_value,
                           int                  dilation_x,
                           int                  dilation_y)
{
    const int kernel_size2 = kernel_width * kernel_height;
    const int x_e          = top_left_x + kernel_width * dilation_x;
    const int y_e          = top_left_y + kernel_height * dilation_y;

    // Linearize volume
    int d = 0;
    // This for loop linearize a volume with 3 slices. This allows:
    // 1) to reduce the iterations of the outer for loop "d"
    // 2) to have an optimized im2col for the first convolution layer where usually we have 3 IFMs
    for (; d <= (kernel_depth - 3); d += 3)
    {
        for (int y = top_left_y; y < y_e; y += dilation_y)
        {
            if ((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
                for (int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    *(out_ptr + 0 * kernel_size2) = pad_value;
                    *(out_ptr + 1 * kernel_size2) = pad_value;
                    *(out_ptr + 2 * kernel_size2) = pad_value;
                }
            }
            else
            {
                for (int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    if ((x < 0 || x >= input_w) && has_pads)
                    {
                        *(out_ptr + 0 * kernel_size2) = pad_value;
                        *(out_ptr + 1 * kernel_size2) = pad_value;
                        *(out_ptr + 2 * kernel_size2) = pad_value;
                    }
                    else
                    {
                        *(out_ptr + 0 * kernel_size2) = *(reinterpret_cast<const T *>(
                            in_ptr + ((d + 0) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                        *(out_ptr + 1 * kernel_size2) = *(reinterpret_cast<const T *>(
                            in_ptr + ((d + 1) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                        *(out_ptr + 2 * kernel_size2) = *(reinterpret_cast<const T *>(
                            in_ptr + ((d + 2) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                    }
                }
            }
        }
        out_ptr += 2 * kernel_size2;
    }

    // Left over
    for (; d < kernel_depth; d++)
    {
        for (int y = top_left_y; y < y_e; y += dilation_y)
        {
            if ((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
                memset(static_cast<void *>(out_ptr), pad_value, kernel_width * sizeof(T));
                out_ptr += kernel_width;
            }
            else
            {
                for (int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    if ((x < 0 || x >= input_w) && has_pads)
                    {
                        *out_ptr = pad_value;
                    }
                    else
                    {
                        *out_ptr = *(reinterpret_cast<const T *>(
                            in_ptr + (d * input_stride_z + y * input_stride_y + x * input_stride_x)));
                    }
                }
            }
        }
    }

    // Append 1 if the convolution layer has biases
    if (has_bias)
    {
        *out_ptr = static_cast<T>(1);
    }
}

template <typename T, bool has_pads>
void linearize_volume_nhwc(const uint8_t *const in_ptr,
                           T                   *out_ptr,
                           bool                 has_bias,
                           int                  start_x,
                           int                  start_y,
                           int                  kernel_width,
                           int                  kernel_height,
                           int                  input_w,
                           int                  input_h,
                           int                  input_c,
                           int                  input_stride_y,
                           int                  input_stride_z,
                           int                  pad_value,
                           int                  dilation_x,
                           int                  dilation_y)
{
    const int end_x        = start_x + kernel_width * dilation_x;
    const int end_y        = start_y + kernel_height * dilation_y;
    const int pad_quant    = kernel_width * input_c;
    const int element_size = static_cast<int>(sizeof(T));
    if ((start_y >= 0) && (end_y < input_h) && (start_x >= 0) && (end_x < input_w) && (dilation_x == 1) &&
        (input_stride_y == input_c * element_size))
    {
        for (int y = start_y; y < end_y; y += dilation_y)
        {
            //optimized for no dilation and no boundary pixels
            memcpy(out_ptr, reinterpret_cast<const T *>(in_ptr + (y * input_stride_z + start_x * input_stride_y)),
                   input_c * kernel_width * element_size);
            out_ptr += input_c * kernel_width;
        }
    }
    else
    {
        for (int y = start_y; y < end_y; y += dilation_y)
        {
            if (y < 0 || y >= input_h)
            {
                memset(static_cast<void *>(out_ptr), pad_value, pad_quant * element_size);
                out_ptr += pad_quant;
            }
            else if (dilation_x > 1 || start_x < 0 || end_x >= input_w || input_stride_y != input_c * element_size)
            {
                for (int x = start_x; x < end_x; x += dilation_x)
                {
                    if (x < 0 || x >= input_w)
                    {
                        memset(static_cast<void *>(out_ptr), pad_value, input_c * element_size);
                        out_ptr += input_c;
                    }
                    else
                    {
                        memcpy(out_ptr, reinterpret_cast<const T *>(in_ptr + (y * input_stride_z + x * input_stride_y)),
                               input_c * element_size);
                        out_ptr += input_c;
                    }
                }
            }
            else
            {
                //optimized for no dilation and no boundary pixels
                memcpy(out_ptr, reinterpret_cast<const T *>(in_ptr + (y * input_stride_z + start_x * input_stride_y)),
                       input_c * kernel_width * element_size);
                out_ptr += input_c * kernel_width;
            }
        }
    }
    // Append 1 if the convolution layer has biases
    if (has_bias)
    {
        *out_ptr = static_cast<T>(1);
    }
}

template <typename T, bool has_pads>
void linearize_volume_nhwc(const uint8_t *const in_ptr,
                           T                   *out_ptr,
                           bool                 has_bias,
                           int                  start_x,
                           int                  start_y,
                           int                  kernel_width,
                           int                  kernel_height,
                           int                  input_w,
                           int                  input_h,
                           int                  input_c,
                           int                  input_stride_y,
                           int                  input_stride_z,
                           int                  pad_value,
                           int                  dilation_x,
                           int                  dilation_y,
                           int                  pad_right)
{
    const int end_x              = start_x + kernel_width * dilation_x;
    const int end_y              = start_y + kernel_height * dilation_y;
    const int pad_quant          = kernel_width * (input_c + pad_right);
    const int element_size       = static_cast<int>(sizeof(T));
    const int channel_chunk_size = input_c * element_size;

    if ((start_y >= 0) && (end_y < input_h) && (start_x >= 0) && (end_x < input_w) && (dilation_x == 1) &&
        (input_stride_y == channel_chunk_size))
    {
        for (int y = start_y; y < end_y; y += dilation_y)
        {
            const uint8_t *offset_ptr = in_ptr + (y * input_stride_z + start_x * input_stride_y);
            for (int e = 0; e < kernel_width; e++)
            {
                memcpy(out_ptr, reinterpret_cast<const T *>(offset_ptr + e * channel_chunk_size), channel_chunk_size);
                out_ptr += input_c + pad_right;
            }
        }
    }
    else
    {
        for (int y = start_y; y < end_y; y += dilation_y)
        {
            if (y < 0 || y >= input_h)
            {
                memset(static_cast<void *>(out_ptr), pad_value, pad_quant * element_size);
                out_ptr += pad_quant;
            }
            else if (dilation_x > 1 || start_x < 0 || end_x >= input_w || input_stride_y != channel_chunk_size)
            {
                for (int x = start_x; x < end_x; x += dilation_x)
                {
                    if (x < 0 || x >= input_w)
                    {
                        memset(static_cast<void *>(out_ptr), pad_value, (input_c + pad_right) * element_size);
                        out_ptr += input_c + pad_right;
                    }
                    else
                    {
                        memcpy(out_ptr, reinterpret_cast<const T *>(in_ptr + (y * input_stride_z + x * input_stride_y)),
                               channel_chunk_size);
                        out_ptr += input_c + pad_right;
                    }
                }
            }
            else
            {
                const uint8_t *offset_ptr = in_ptr + (y * input_stride_z + start_x * input_stride_y);
                for (int e = 0; e < kernel_width; e++)
                {
                    memcpy(out_ptr, reinterpret_cast<const T *>(offset_ptr + e * channel_chunk_size),
                           channel_chunk_size);
                    out_ptr += input_c + pad_right;
                }
            }
        }
    }
    // Append 1 if the convolution layer has biases
    if (has_bias)
    {
        *out_ptr = static_cast<T>(1);
    }
}

template <typename T, bool has_pads, bool is_nchw>
void run_im2col(const ITensor                        *src,
                ITensor                              *dst,
                const Window                         &window,
                DataLayout                            data_layout,
                const PadStrideInfo                  &conv_info,
                std::pair<unsigned int, unsigned int> convolved_dims,
                const Size2D                         &kernel_dims,
                const Size2D                         &dilation,
                uint32_t                              input_pad_right,
                bool                                  has_bias)
{
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const int input_w        = src->info()->dimension(width_idx);
    const int input_h        = src->info()->dimension(height_idx);
    const int input_c        = src->info()->dimension(channel_idx);
    const int input_stride_x = src->info()->strides_in_bytes().x();
    const int input_stride_y = src->info()->strides_in_bytes().y();
    const int input_stride_z = src->info()->strides_in_bytes().z();
    const int pad_left       = conv_info.pad_left();
    const int pad_top        = conv_info.pad_top();
    const int stride_x       = conv_info.stride().first;
    const int stride_y       = conv_info.stride().second;
    const int pad_value =
        is_data_type_quantized(src->info()->data_type()) ? src->info()->quantization_info().uniform().offset : 0;

    const auto kernel_width  = kernel_dims.width;
    const auto kernel_height = kernel_dims.height;

    Window window_in_out(window);
    // The first three dimensions of the input and output are increased by the inner loops
    window_in_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Create iterators
    Iterator in(src, window_in_out);
    Iterator out(dst, window_in_out);

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const int start_w = id[width_idx] * stride_x - pad_left;
            const int start_h = id[height_idx] * stride_y - pad_top;

            // Get pointers
            const uint8_t *const input_ptr = in.ptr();
            auto                 output_ptr =
                reinterpret_cast<T *>(out.ptr() + (id[width_idx] + id[height_idx] * convolved_dims.first) *
                                                      dst->info()->strides_in_bytes().y());

            // Linearize volume
            if (is_nchw)
            {
                linearize_volume_nchw<T, has_pads>(
                    input_ptr, output_ptr, has_bias, start_w, start_h, kernel_width, kernel_height, input_c, input_w,
                    input_h, input_stride_x, input_stride_y, input_stride_z, pad_value, dilation.x(), dilation.y());
            }
            else
            {
                if (input_pad_right > 0)
                {
                    linearize_volume_nhwc<T, has_pads>(input_ptr, output_ptr, has_bias, start_w, start_h, kernel_width,
                                                       kernel_height, input_w, input_h, input_c, input_stride_y,
                                                       input_stride_z, pad_value, dilation.x(), dilation.y(),
                                                       input_pad_right);
                }
                else
                {
                    linearize_volume_nhwc<T, has_pads>(input_ptr, output_ptr, has_bias, start_w, start_h, kernel_width,
                                                       kernel_height, input_w, input_h, input_c, input_stride_y,
                                                       input_stride_z, pad_value, dilation.x(), dilation.y());
                }
            }
        },
        in, out);
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_DIRECTCONV2D_IMPL_H
