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

template <bool Specialized, int PadBottom=0, int PadRight=0>
void winograd_output_transform_2x2_3x3_fp32_process_tile(
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
  constexpr int OutputTileRows = 2, OutputTileCols = 2;
  const int pad_bottom = Specialized ? PadBottom : _pad_bottom;
  const int pad_right = Specialized ? PadRight : _pad_right;

  const int cells_i = OutputTileRows - pad_bottom;
  const int cells_j = OutputTileCols - pad_right;

  // Construct a map to the output cells
  float *outptrs[OutputTileRows][OutputTileCols];
  for (int i = 0; i < cells_i; i++)
  {
    for (int j = 0; j < cells_j; j++)
    {
      outptrs[i][j] = output + i*output_row_stride + j*output_col_stride;
    }
  }
  const float *inptr = matrix_base;
  const float *bptr = biases;

  if (bptr)
  {
    // For each channel of the output
    int channels_remaining = n_channels;
#ifdef __aarch64__
    for (; channels_remaining >= 4; channels_remaining -= 4)
    {
      // Matrices used and computed during this transform
      float32x4_t F[4][4], FZ[4][2], f[2][2], b;

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

      // Load the bias vector
      b = vld1q_f32(bptr);
      bptr += 4;

      // Write out the output tile
      for (int i = 0; i < cells_i; i++)
      {
        for (int j = 0; j < cells_j; j++)
        {
          vst1q_f32(outptrs[i][j], vaddq_f32(f[i][j], b));
          outptrs[i][j] += 4;
        }
      }
    }
#endif  // __aarch64__
#ifdef __arm_any__
    for (; channels_remaining >= 2; channels_remaining -= 2)
    {
      // Matrices used and computed during this transform
      float32x2_t F[4][4], FZ[4][2], f[2][2], b;

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

      // Load the bias vector
      b = vld1_f32(bptr);
      bptr += 2;

      // Write out the output tile
      for (int i = 0; i < cells_i; i++)
      {
        for (int j = 0; j < cells_j; j++)
        {
          vst1_f32(outptrs[i][j], vadd_f32(f[i][j], b));
          outptrs[i][j] += 2;
        }
      }
    }
#endif  // __arm_any__
    for (; channels_remaining; channels_remaining--)
    {
      // Matrices used and computed during this transform
      float F[4][4], FZ[4][2], f[2][2], b;

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

      // Load the bias
      b = *(bptr++);

      // Write out the output tile
      for (int i = 0; i < cells_i; i++)
      {
        for (int j = 0; j < cells_j; j++)
        {
          *(outptrs[i][j]++) = f[i][j] + b;
        }
      }
    }
  }
  else
  {
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
}

}  // namespace (anonymous)

namespace winograd
{
using Tiles = OutputTransformImplTiles<3, 3, 4, 4, float>;

template <>
const Tiles::TileFn Tiles::tilefn_generic = winograd_output_transform_2x2_3x3_fp32_process_tile<false>;

template <>
const Tiles::TileFn Tiles::tilefn_unpadded = winograd_output_transform_2x2_3x3_fp32_process_tile<true>;

template <>
const Tiles::TileFn Tiles::tilefn_bottom_padded[n_pad_bottom] = {
  winograd_output_transform_2x2_3x3_fp32_process_tile<true, 1, 0>
};

template <>
const Tiles::TileFn Tiles::tilefn_right_padded[n_pad_right] = {
  winograd_output_transform_2x2_3x3_fp32_process_tile<true, 0, 1>
};

template class OutputTransform<3, 3, 4, 4, float>;
}  // namespace winograd

