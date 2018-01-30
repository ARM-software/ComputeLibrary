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
// Partial specialisation for FP32 to FP32
template <int OutputTileRows, int OutputTileCols,
          int KernelRows, int KernelCols,
          int StrideRows, int StrideCols>
struct DepthwiseConvolutionImpl<OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols, float, float>
{
  typedef DepthwiseConvolution<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    float, float
  > DWC;

  template <
    int in_pad_top, int in_pad_left, int in_pad_bottom, int in_pad_right,
    int out_pad_bottom, int out_pad_right
  >
  static void process_tile(
    const int n_channels,
    const float* const weights,
    const float* const inptr,
    const int in_row_stride,
    const int in_col_stride,
    float* const outptr,
    const int out_row_stride,
    const int out_col_stride
  );
};


template <int OTR, int OTC, int KR, int KC, int SR, int SC>
template <
  int in_pad_top, int in_pad_left, int in_pad_bottom, int in_pad_right,
  int out_pad_bottom, int out_pad_right
>
void DepthwiseConvolutionImpl<OTR, OTC, KR, KC, SR, SC, float, float>::process_tile(
  const int n_channels,
  const float* const weights,
  const float* const inptr,
  const int in_row_stride,
  const int in_col_stride,
  float* const outptr,
  const int out_row_stride,
  const int out_col_stride
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

  // Compute valid ranges of the tile
  constexpr int in_cells_i = inner_tile_rows - in_pad_bottom;
  constexpr int in_cells_j = inner_tile_cols - in_pad_right;
  constexpr int out_cells_i = output_tile_rows - out_pad_bottom;
  constexpr int out_cells_j = output_tile_cols - out_pad_right;

  // Instantiate pointers
  const float* inptr_base = inptr;
  const float* wptr_base = weights;
  float* outptr_base = outptr;

  const int weight_col_stride = n_channels;
  const int weight_row_stride = kernel_cols * n_channels;

  // Perform the depthwise convolution
  int channels_remaining = n_channels;
#ifdef __aarch64__
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Load input tile
    float32x4_t u[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      const float* const inptr_row = inptr_base + (i - in_pad_top)*in_row_stride;
      for (int j = 0; j < inner_tile_cols; j++)
      {
        if (i < in_pad_top || in_cells_i <= i ||
            j < in_pad_left || in_cells_j <= j)
        {
          u[i][j] = vdupq_n_f32(0.0f);
        }
        else
        {
          u[i][j] = vld1q_f32(inptr_row + (j - in_pad_left)*in_col_stride);
        }
      }
    }
    inptr_base += 4;

    // Load weights tile
    float32x4_t w[kernel_rows][kernel_cols];
    for (int i = 0; i < kernel_rows; i++)
    {
      const float* const wptr_row = wptr_base + i*weight_row_stride;
      for (int j = 0; j < kernel_cols; j++)
      {
        w[i][j] = vld1q_f32(wptr_row + j*weight_col_stride);
      }
    }
    wptr_base += 4;

    // Perform the convolution
    float32x4_t v[out_cells_i][out_cells_j];
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
              v[out_i][out_j] = vmulq_f32(w[in_i][in_j], u[i][j]);
            }
            else
            {
              // v[out_i][out_j] += w[in_i][in_j] * u[i][j];
              v[out_i][out_j] = vmlaq_f32(v[out_i][out_j], w[in_i][in_j], u[i][j]);
            }
          }
        }
      }
    }

    // Store the output tile
    for (int i = 0; i < out_cells_i; i++)
    {
      float* const outptr_row = outptr_base + i*out_row_stride;
      for (int j = 0; j < out_cells_j; j++)
      {
        vst1q_f32(outptr_row + j*out_col_stride, v[i][j]);
      }
    }
    outptr_base += 4;
  }
#endif  // __aarch64__
  for (; channels_remaining; channels_remaining--)
  {
    // Load input tile
    float u[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      const float* const inptr_row = inptr_base + (i - in_pad_top)*in_row_stride;
      for (int j = 0; j < inner_tile_cols; j++)
      {
        if (i < in_pad_top || in_cells_i <= i ||
            j < in_pad_left || in_cells_j <= j)
        {
          u[i][j] = static_cast<float>(0);
        }
        else
        {
          u[i][j] = *(inptr_row + (j - in_pad_left)*in_col_stride);
        }
      }
    }
    inptr_base++;

    // Load weights tile
    float w[kernel_rows][kernel_cols];
    for (int i = 0; i < kernel_rows; i++)
    {
      const float* const wptr_row = wptr_base + i*weight_row_stride;
      for (int j = 0; j < kernel_cols; j++)
      {
        w[i][j] = *(wptr_row + j*weight_col_stride);
      }
    }
    wptr_base++;

    // Perform the convolution
    float v[out_cells_i][out_cells_j];
    for (int out_i = 0; out_i < out_cells_i; out_i++)
    {
      for (int out_j = 0; out_j < out_cells_j; out_j++)
      {
        // Clear the accumulator
        v[out_i][out_j] = static_cast<float>(0);

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
      float* const outptr_row = outptr_base + i*out_row_stride;
      for (int j = 0; j < out_cells_j; j++)
      {
        *(outptr_row + j*out_col_stride) = v[i][j];
      }
    }
    outptr_base++;
  }
}

}  // namespace depthwise
