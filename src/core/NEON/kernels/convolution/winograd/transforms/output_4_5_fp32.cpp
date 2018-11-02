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

#include "arm_compute/core/NEON/kernels/convolution/winograd/transforms/output.hpp"
#include "arm_compute/core/NEON/kernels/convolution/winograd/winograd_output_transform.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/arm.hpp"

namespace
{

template <bool Specialized, int PadRight=0>
void winograd_output_transform_4_5_fp32_process_tile(
  const int n_channels,
  const float* const matrix_base,
  const int matrix_stride,
  const float* const biases,
  float* const output,
  const int output_row_stride,
  const int output_col_stride,
  const int _pad_bottom,
  const int _pad_right
)
{
  (void) output_row_stride;
  (void) _pad_bottom;
  constexpr int output_tile_cols = 4;
  constexpr int inner_tile_cols = 8;

  const int pad_right = Specialized ? PadRight : _pad_right;
  const int cells_j = output_tile_cols - pad_right;

  // Construct a map to the output cells
  float *outptrs[cells_j];
  for (int j = 0; j < cells_j; j++)
  {
    outptrs[j] = output + j*output_col_stride;
  }
  const float *inptr = matrix_base;
  const float *bptr = biases;

  // For each channel of the output
  int channels_remaining = n_channels;
#ifdef __arm_any__
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used and computed during this transform
    float32x4_t F[inner_tile_cols], f[output_tile_cols], b = vdupq_n_f32(0.0f);

    // Read a 1x8 tile in the Winograd domain
    for (int j = 0; j < inner_tile_cols; j++)
    {
      F[j] = vld1q_f32(inptr + j*matrix_stride);
    }
    inptr += 4;

    f[0] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[6], 1), F[5], 1), F[4], 1), F[3], 1), F[2], 1), F[1], 1), F[0], 1);
    f[1] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[2], 1), F[6], 3), F[4], 2), F[3], -2), F[5], -3), F[1], -1);
    f[2] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[2], 1), F[1], 1), F[6], 9), F[5], 9), F[4], 4), F[3], 4);
    f[3] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[7], 1), F[2], 1), F[6], 27), F[4], 8), F[3], -8), F[5], -27), F[1], -1);

    // Write out the output tile
    if (bptr != 0)
    {
      b = vld1q_f32(bptr);
      bptr += 4;
    }
    for (int j = 0; j < cells_j; j++)
    {
      vst1q_f32(outptrs[j], f[j] + b);
      outptrs[j] += 4;
    }
  }
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    // Matrices used and computed during this transform
    float32x2_t F[inner_tile_cols], f[output_tile_cols], b = vdup_n_f32(0.0f);

    // Read a 1x8 tile in the Winograd domain
    for (int j = 0; j < inner_tile_cols; j++)
    {
      F[j] = vld1_f32(inptr + j*matrix_stride);
    }
    inptr += 2;

    f[0] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[6], 1), F[5], 1), F[4], 1), F[3], 1), F[2], 1), F[1], 1), F[0], 1);
    f[1] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[2], 1), F[6], 3), F[4], 2), F[3], -2), F[5], -3), F[1], -1);
    f[2] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[2], 1), F[1], 1), F[6], 9), F[5], 9), F[4], 4), F[3], 4);
    f[3] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[7], 1), F[2], 1), F[6], 27), F[4], 8), F[3], -8), F[5], -27), F[1], -1);

    // Write out the output tile
    if (bptr != 0)
    {
      b = vld1_f32(bptr);
      bptr += 2;
    }
    for (int j = 0; j < cells_j; j++)
    {
      vst1_f32(outptrs[j], f[j] + b);
      outptrs[j] += 2;
    }
  }
#endif  // __arm_any__
  for (; channels_remaining; channels_remaining--)
  {
    // Matrices used and computed during this transform
    float F[inner_tile_cols], f[output_tile_cols], b = 0.0f;

    // Read a 1x8 tile in the Winograd domain
    for (int j = 0; j < inner_tile_cols; j++)
    {
      F[j] = *(inptr + j*matrix_stride);
    }
    inptr++;

    f[0] = F[0]*1 + F[1]*1 + F[2]*1 + F[3]*1 + F[4]*1 + F[5]*1 + F[6]*1;
    f[1] = F[1]*-1 + F[5]*-3 + F[3]*-2 + F[4]*2 + F[6]*3 + F[2]*1;
    f[2] = F[3]*4 + F[4]*4 + F[5]*9 + F[6]*9 + F[1]*1 + F[2]*1;
    f[3] = F[1]*-1 + F[5]*-27 + F[3]*-8 + F[4]*8 + F[6]*27 + F[2]*1 + F[7]*1;

    // Write out the output tile
    if (bptr != 0)
    {
      b = *(bptr++);
    }
    for (int j = 0; j < cells_j; j++)
    {
      *(outptrs[j]++) = f[j] + b;
    }
  }
}

}  // namespace (anonymous)

namespace winograd
{
using Tiles = OutputTransformImplTiles<1, 5, 1, 8, float>;

template <>
const Tiles::TileFn Tiles::tilefn_unpadded = winograd_output_transform_4_5_fp32_process_tile<true>;

template <>
const Tiles::TileFn Tiles::tilefn_right_padded[n_pad_right] = {
  winograd_output_transform_4_5_fp32_process_tile<true, 1>,
  winograd_output_transform_4_5_fp32_process_tile<true, 2>,
  winograd_output_transform_4_5_fp32_process_tile<true, 3>
};

template class OutputTransform<1, 5, 1, 8, float>;
template class OutputTransform<5, 1, 8, 1, float>;
}  // namespace winograd
