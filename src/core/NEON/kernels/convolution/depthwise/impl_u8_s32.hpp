/*
 * Copyright (c) 2018 ARM Limited.
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

/*
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 *          NOTE: Header to be included by implementation files only.
 *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

#include "arm_compute/core/NEON/kernels/convolution/common/arm.hpp"
#include "arm_compute/core/NEON/kernels/convolution/depthwise/impl_base.hpp"

#pragma once

namespace depthwise
{
// Partial specialisation for U8 to S32
template <int OutputTileRows, int OutputTileCols,
        int KernelRows, int KernelCols,
        int StrideRows, int StrideCols>
struct DepthwiseConvolutionImpl<OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols, uint8_t, int32_t>
{
    typedef DepthwiseConvolution<
            OutputTileRows, OutputTileCols,
            KernelRows, KernelCols,
            StrideRows, StrideCols,
            uint8_t, int32_t
    > DWC;

    template <
            bool Specialize=false,  // Specialize (or not) the method
            int InPadTop=0,         // If specialized, top padding
            int InPadLeft=0,        // If specialized, left padding
            int InPadBottom=0,      // If specialized, bottom padding
            int InPadRight=0,       // If specialized, right padding
            int OutPadBottom=0,     // If specialized, bottom output padding
            int OutPadRight=0       // If specialized, bottom right padding
    >
    static void process_tile(
            const int n_channels,
            const uint8_t* const weights,
            const int weight_row_stride,
            const int weight_col_stride,
            const uint8_t* const inptr,
            const int in_row_stride,
            const int in_col_stride,
            int32_t* const outptr,
            const int out_row_stride,
            const int out_col_stride,
            const int in_pad_top=0,
            const int in_pad_left=0,
            const int in_pad_bottom=0,
            const int in_pad_right=0,
            const int out_pad_bottom=0,
            const int out_pad_right=0,
            const int input_offset=0,
            const int weights_offset=0);
};


template <int OTR, int OTC, int KR, int KC, int SR, int SC>
template <
        bool Specialize,
        int InPadTop, int InPadLeft, int InPadBottom, int InPadRight,
        int OutPadBottom, int OutPadRight
>
void DepthwiseConvolutionImpl<OTR, OTC, KR, KC, SR, SC, uint8_t, int32_t>::process_tile(
        const int n_channels,
        const uint8_t *__restrict__ const weights,
        const int weight_row_stride,
        const int weight_col_stride,
        const uint8_t *__restrict__ const inptr,
        const int in_row_stride,
        const int in_col_stride,
        int32_t *__restrict__ const outptr,
        const int out_row_stride,
        const int out_col_stride,
        const int _in_pad_top,
        const int _in_pad_left,
        const int _in_pad_bottom,
        const int _in_pad_right,
        const int _out_pad_bottom,
        const int _out_pad_right,
        const int _input_offset,
        const int _weights_offset
)
{
    constexpr auto inner_tile_rows = DWC::inner_tile_rows;
    constexpr auto inner_tile_cols = DWC::inner_tile_cols;
    constexpr auto kernel_rows = DWC::kernel_rows;
    constexpr auto kernel_cols = DWC::kernel_cols;
    constexpr auto output_tile_rows = DWC::output_tile_rows;
    constexpr auto output_tile_cols = DWC::output_tile_cols;
    constexpr auto stride_rows = DWC::stride_rows;
    constexpr auto stride_cols = DWC::stride_cols;

    // Extract parameters
    const int in_pad_top = Specialize ? InPadTop : _in_pad_top;
    const int in_pad_left = Specialize ? InPadLeft : _in_pad_left;
    const int in_pad_bottom = Specialize ? InPadBottom : _in_pad_bottom;
    const int in_pad_right = Specialize ? InPadRight : _in_pad_right;
    const int out_pad_bottom = Specialize ? OutPadBottom : _out_pad_bottom;
    const int out_pad_right = Specialize ? OutPadRight : _out_pad_right;

    // Compute valid ranges of the tile
    const int in_cells_i = inner_tile_rows - in_pad_bottom;
    const int in_cells_j = inner_tile_cols - in_pad_right;
    const int out_cells_i = output_tile_rows - out_pad_bottom;
    const int out_cells_j = output_tile_cols - out_pad_right;

    // Instantiate pointers
    const uint8_t* __restrict__ inptr_base = inptr;
    const uint8_t* __restrict__ wptr_base = weights;
    int32_t* __restrict__ outptr_base = outptr;

    // Perform the depthwise convolution
    int channels_remaining = n_channels;
#ifdef __aarch64__
    const int32x4_t v_input_offset = vdupq_n_s32(_input_offset);
    const int32x4_t v_weights_offset = vdupq_n_s32(_weights_offset);
    for (; channels_remaining >= 16; channels_remaining -= 16)
    {
        // Load input tile
        int32x4x4_t u[inner_tile_rows][inner_tile_cols];
        for (int i = 0; i < inner_tile_rows; i++)
        {
            const uint8_t* const inptr_row = inptr_base + (i - in_pad_top)*in_row_stride;
            for (int j = 0; j < inner_tile_cols; j++)
            {
                if (i < in_pad_top || in_cells_i <= i ||
                    j < in_pad_left || in_cells_j <= j)
                {
                    u[i][j].val[0] = vdupq_n_s32(0);
                    u[i][j].val[1] = vdupq_n_s32(0);
                    u[i][j].val[2] = vdupq_n_s32(0);
                    u[i][j].val[3] = vdupq_n_s32(0);
                }
                else
                {
                    const uint8x16_t uv = vld1q_u8(inptr_row + (j - in_pad_left)*in_col_stride);
                    u[i][j].val[0] = vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_low_u8(uv)))));
                    u[i][j].val[1] = vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_low_u8(uv)))));
                    u[i][j].val[2] = vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_high_u8(uv)))));
                    u[i][j].val[3] = vaddw_s16(v_input_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_high_u8(uv)))));
                }
            }
        }
        inptr_base += 16;

        // Load weights tile
        int32x4x4_t w[kernel_rows][kernel_cols];
        for (int i = 0; i < kernel_rows; i++)
        {
            const uint8_t* const wptr_row = wptr_base + i*weight_row_stride;
            for (int j = 0; j < kernel_cols; j++)
            {
                const uint8x16_t wv = vld1q_u8(wptr_row + j*weight_col_stride);
                w[i][j].val[0] = vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_low_u8(wv)))));
                w[i][j].val[1] = vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_low_u8(wv)))));
                w[i][j].val[2] = vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vget_high_u8(wv)))));
                w[i][j].val[3] = vaddw_s16(v_weights_offset, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(vget_high_u8(wv)))));
            }
        }
        wptr_base += 16;

        // Perform the convolution
        int32x4x4_t v[output_tile_rows][output_tile_cols];
        for (int out_i = 0; out_i < out_cells_i; out_i++)
        {
            for (int out_j = 0; out_j < out_cells_j; out_j++)
            {
                // Base co-ordinate
                const int base_i = out_i * stride_rows;
                const int base_j = out_j * stride_cols;

                // Fill the accumulator
                for (int in_i = 0; in_i < kernel_rows; in_i++)
                {
                    const int i = base_i + in_i;
                    for (int in_j = 0; in_j < kernel_cols; in_j++)
                    {
                        const int j = base_j + in_j;
                        if (in_i == 0 && in_j == 0)
                        {
                            // v[out_i][out_j] = w[in_i][in_j] * u[i][j];
                            v[out_i][out_j].val[0] = vmulq_s32(w[in_i][in_j].val[0], u[i][j].val[0]);
                            v[out_i][out_j].val[1] = vmulq_s32(w[in_i][in_j].val[1], u[i][j].val[1]);
                            v[out_i][out_j].val[2] = vmulq_s32(w[in_i][in_j].val[2], u[i][j].val[2]);
                            v[out_i][out_j].val[3] = vmulq_s32(w[in_i][in_j].val[3], u[i][j].val[3]);
                        }
                        else
                        {
                            // v[out_i][out_j] += w[in_i][in_j] * u[i][j];
                            v[out_i][out_j].val[0] = vmlaq_s32(v[out_i][out_j].val[0], w[in_i][in_j].val[0], u[i][j].val[0]);
                            v[out_i][out_j].val[1] = vmlaq_s32(v[out_i][out_j].val[1], w[in_i][in_j].val[1], u[i][j].val[1]);
                            v[out_i][out_j].val[2] = vmlaq_s32(v[out_i][out_j].val[2], w[in_i][in_j].val[2], u[i][j].val[2]);
                            v[out_i][out_j].val[3] = vmlaq_s32(v[out_i][out_j].val[3], w[in_i][in_j].val[3], u[i][j].val[3]);
                        }
                    }
                }
            }
        }

        // Store the output tile
        for (int i = 0; i < out_cells_i; i++)
        {
            int32_t* const outptr_row = outptr_base + i*out_row_stride;
            for (int j = 0; j < out_cells_j; j++)
            {
                vst1q_s32(outptr_row + j*out_col_stride, v[i][j].val[0]);
                vst1q_s32(outptr_row + j*out_col_stride + 4, v[i][j].val[1]);
                vst1q_s32(outptr_row + j*out_col_stride + 8, v[i][j].val[2]);
                vst1q_s32(outptr_row + j*out_col_stride + 12, v[i][j].val[3]);
            }
        }
        outptr_base += 16;
    }
#endif  // __aarch64__
    for (; channels_remaining; channels_remaining--)
    {
        // Load input tile
        int32_t u[inner_tile_rows][inner_tile_cols];
        for (int i = 0; i < inner_tile_rows; i++)
        {
            const uint8_t* const inptr_row = inptr_base + (i - in_pad_top)*in_row_stride;
            for (int j = 0; j < inner_tile_cols; j++)
            {
                if (i < in_pad_top || in_cells_i <= i ||
                    j < in_pad_left || in_cells_j <= j)
                {
                    u[i][j] = static_cast<uint8_t>(0);
                }
                else
                {
                    u[i][j] = static_cast<int32_t >(*(inptr_row + (j - in_pad_left)*in_col_stride)) + _input_offset;
                }
            }
        }
        inptr_base++;

        // Load weights tile
        int32_t w[kernel_rows][kernel_cols];
        for (int i = 0; i < kernel_rows; i++)
        {
            const uint8_t* const wptr_row = wptr_base + i*weight_row_stride;
            for (int j = 0; j < kernel_cols; j++)
            {
                w[i][j] = static_cast<int32_t >(*(wptr_row + j*weight_col_stride)) + _weights_offset;
            }
        }
        wptr_base++;

        // Perform the convolution
        int32_t v[output_tile_rows][output_tile_cols];
        for (int out_i = 0; out_i < out_cells_i; out_i++)
        {
            for (int out_j = 0; out_j < out_cells_j; out_j++)
            {
                // Clear the accumulator
                v[out_i][out_j] = static_cast<int32_t>(0);

                // Base co-ordinate
                const int base_i = out_i * stride_rows;
                const int base_j = out_j * stride_cols;

                // Fill the accumulator
                for (int in_i = 0; in_i < kernel_rows; in_i++)
                {
                    const int i = base_i + in_i;
                    for (int in_j = 0; in_j < kernel_cols; in_j++)
                    {
                        const int j = base_j + in_j;
                        v[out_i][out_j] += w[in_i][in_j] * u[i][j];
                    }
                }
            }
        }

        // Store the output tile
        for (int i = 0; i < out_cells_i; i++)
        {
            int32_t* const outptr_row = outptr_base + i*out_row_stride;
            for (int j = 0; j < out_cells_j; j++)
            {
                *(outptr_row + j*out_col_stride) = v[i][j];
            }
        }
        outptr_base++;
    }
}

}  // namespace depthwise
