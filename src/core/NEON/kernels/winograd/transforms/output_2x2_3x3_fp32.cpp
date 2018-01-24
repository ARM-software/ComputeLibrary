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

#include "transforms/output.hpp"
#include "winograd_gemm.hpp"
#include "arm.hpp"

namespace winograd
{

using Transform = WinogradGEMM<2, 2, 3, 3>::OutputTransform<float>;

template <>
template <>
int Transform::ops_performed(const Tensor4DShape &shape)
{
  // NOTE: Cost in FLOPs rather than instructions or uops.
  const int tile_M = iceildiv(shape.n_rows, 2);
  const int tile_N = iceildiv(shape.n_cols, 2);
  return 24 * tile_M * tile_N * shape.n_channels;
}

/* F(2x2, 3x3) constructs 2x2 output tiles from a 3x3 convolution. Since we use
 * enough tiles to cover the output space each output tile may contain 0 or 1
 * padded values to the right and bottom columns or rows of the tile, e.g.:
 *
 *      ___     ___
 *     |   |   |  X|
 *     |___|   |__X|
 *
 *      ___     ___
 *     |   |   |  X|
 *     |X_X|   |X_X|
 *
 *
 * We provide a specialised output transform for each of these instances.
 * Consequently we below construct an array of the various padding options, the
 * array contains pointers to the specific implementations.
 */
template <>
template <>
template <int pad_bottom, int pad_right>
void Transform::process_tile(
  const int n_channels,
  const float* const matrix_base,
  const int matrix_stride,
  float* const output,
  const int output_row_stride,
  const int output_col_stride
)
{
  constexpr int cells_i = 2 - pad_bottom;
  constexpr int cells_j = 2 - pad_right;

  // Construct a map to the output cells
  float *outptrs[cells_i][cells_j];
  for (int i = 0; i < cells_i; i++)
  {
    for (int j = 0; j < cells_j; j++)
    {
      outptrs[i][j] = output + i*output_row_stride + j*output_col_stride;
    }
  }
  const float *inptr = matrix_base;

  // For each channel of the output
  int channels_remaining = n_channels;
#ifdef __aarch64__
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used and computed during this transform
    float32x4_t F[4][4], FZ[4][2], f[2][2];

    // Read a 4x4 tile in the Winograd domain
    for (int i = 0, m = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++, m++)
      {
        F[i][j] = vld1q_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 4;

    // Compute the matrix F Z
    for (int i = 0; i < 4; i++)
    {
      // FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
      FZ[i][0] = vaddq_f32(vaddq_f32(F[i][0], F[i][1]), F[i][2]);

      // FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
      FZ[i][1] = vsubq_f32(vsubq_f32(F[i][1], F[i][2]), F[i][3]);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 2; j++)
    {
      // f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
      f[0][j] = vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), FZ[2][j]);

      // f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
      f[1][j] = vsubq_f32(vsubq_f32(FZ[1][j], FZ[2][j]), FZ[3][j]);
    }

    // Write out the output tile
    for (int i = 0; i < cells_i; i++)
    {
      for (int j = 0; j < cells_j; j++)
      {
        vst1q_f32(outptrs[i][j], f[i][j]);
        outptrs[i][j] += 4;
      }
    }
  }
#endif  // __aarch64__
#ifdef __arm_any__
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    // Matrices used and computed during this transform
    float32x2_t F[4][4], FZ[4][2], f[2][2];

    // Read a 4x4 tile in the Winograd domain
    for (int i = 0, m = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++, m++)
      {
        F[i][j] = vld1_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 2;

    // Compute the matrix F Z
    for (int i = 0; i < 4; i++)
    {
      // FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
      FZ[i][0] = vadd_f32(vadd_f32(F[i][0], F[i][1]), F[i][2]);

      // FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
      FZ[i][1] = vsub_f32(vsub_f32(F[i][1], F[i][2]), F[i][3]);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 2; j++)
    {
      // f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
      f[0][j] = vadd_f32(vadd_f32(FZ[0][j], FZ[1][j]), FZ[2][j]);

      // f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
      f[1][j] = vsub_f32(vsub_f32(FZ[1][j], FZ[2][j]), FZ[3][j]);
    }

    // Write out the output tile
    for (int i = 0; i < cells_i; i++)
    {
      for (int j = 0; j < cells_j; j++)
      {
        vst1_f32(outptrs[i][j], f[i][j]);
        outptrs[i][j] += 2;
      }
    }
  }
#endif  // __arm_any__
  for (; channels_remaining; channels_remaining--)
  {
    // Matrices used and computed during this transform
    float F[4][4], FZ[4][2], f[2][2];

    // Read a 4x4 tile in the Winograd domain
    for (int i = 0, m = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++, m++)
      {
        F[i][j] = *(inptr + m*matrix_stride);
      }
    }
    inptr++;

    // Compute the matrix F Z
    for (int i = 0; i < 4; i++)
    {
      FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
      FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 2; j++)
    {
      f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
      f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
    }

    // Write out the output tile
    for (int i = 0; i < cells_i; i++)
    {
      for (int j = 0; j < cells_j; j++)
      {
        *(outptrs[i][j]++) = f[i][j];
      }
    }
  }
}

template <>
template <>
const Transform::TileFn Transform::tile_fns[max_pad_bottom][max_pad_right] =
{
  {
    Transform::template process_tile<0, 0>,  // No padding
    Transform::template process_tile<0, 1>,  // Right padding
  },
  {
    Transform::template process_tile<1, 0>,  // Bottom padding
    Transform::template process_tile<1, 1>,  // Bottom and right padding
  }
};

template struct WinogradGEMM<2, 2, 3, 3>::OutputTransform<float>;
}  // namespace winograd
