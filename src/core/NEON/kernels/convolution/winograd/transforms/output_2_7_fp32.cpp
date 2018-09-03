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
#include "arm_compute/core/NEON/kernels/convolution/winograd/winograd_gemm.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/arm.hpp"

namespace winograd
{

using Transform = WinogradGEMM<1, 2, 1, 7>::OutputTransform<float>;
using TransformTransposed = WinogradGEMM<2, 1, 7, 1>::OutputTransform<float>;

template <>
template <>
int Transform::ops_performed(const Tensor4DShape &shape)
{
  (void) shape;
  return 0;  // TODO
}

template <>
template <>
template <int pad_bottom, int pad_right>
void Transform::process_tile(
  const int n_channels,
  const float* const matrix_base,
  const int matrix_stride,
  const float* const biases,
  float* const output,
  const int output_row_stride,
  const int output_col_stride
)
{
  (void) output_row_stride;
  constexpr int cells_j = output_tile_cols - pad_right;

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
    f[1] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[7], 1), F[2], 1), F[6], 3), F[4], 2), F[3], -2), F[5], -3), F[1], -1);

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
    f[1] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[7], 1), F[2], 1), F[6], 3), F[4], 2), F[3], -2), F[5], -3), F[1], -1);

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
    f[1] = F[1]*-1 + F[5]*-3 + F[3]*-2 + F[4]*2 + F[6]*3 + F[2]*1 + F[7]*1;

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

template <>
template <>
const Transform::TileFn Transform::tile_fns[max_pad_bottom][max_pad_right] =
{
  {
    Transform::template process_tile<0, 0>,
    Transform::template process_tile<0, 1>,
  },
};


template <>
template <>
const TransformTransposed::TileFn TransformTransposed::tile_fns[max_pad_bottom][max_pad_right] = {};

template struct WinogradGEMM<1, 2, 1, 7>::OutputTransform<float>;
template struct WinogradGEMM<2, 1, 7, 1>::OutputTransform<float>;
}  // namespace winograd
