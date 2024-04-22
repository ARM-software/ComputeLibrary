/*
 * Copyright (c) 2022 Arm Limited.
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

#include <algorithm>
#include <cstddef>
#include <arm_neon.h>

namespace arm_conv {
namespace winograd {
namespace output_transform {

void arm_fp32_2x2_3x3(
  unsigned int n_channels,
  const float* inptr,
  const size_t matrix_stride,
  const float* bptr,
  float *outptr,
  const size_t output_row_stride,
  const size_t output_col_stride,
  const float output_min,
  const float output_max
)
{
  constexpr auto output_tile_rows = 2u, output_tile_cols = 2u;

  // For each channel of the output
  for (; n_channels >= 4; n_channels -= 4)
  {
    // Matrices used and computed during this transform
    float32x4_t F[4][4], FZ[4][2], f[2][2], b;

    // Read a 4x4 tile in the Winograd domain
    for (auto i = 0u, m = 0u; i < 4; i++)
    {
      for (auto j = 0u; j < 4; j++, m++)
      {
        F[i][j] = vld1q_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 4;

    // Compute the matrix F Z
    for (auto i = 0u; i < 4; i++)
    {
      // FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
      FZ[i][0] = vaddq_f32(vaddq_f32(F[i][0], F[i][1]), F[i][2]);

      // FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
      FZ[i][1] = vsubq_f32(vsubq_f32(F[i][1], F[i][2]), F[i][3]);
    }

    // Compute the output tile f = ZT F Z
    for (auto j = 0u; j < 2; j++)
    {
      // f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
      f[0][j] = vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), FZ[2][j]);

      // f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
      f[1][j] = vsubq_f32(vsubq_f32(FZ[1][j], FZ[2][j]), FZ[3][j]);
    }

    // Load the bias vector
    if (bptr != nullptr)
    {
      b = vld1q_f32(bptr);
      bptr += 4;
    }
    else
    {
      b = vdupq_n_f32(0.0f);
    }

    // Write out the output tile
    for (auto i = 0u; i < output_tile_rows; i++)
    {
      for (auto j = 0u; j < output_tile_cols; j++)
      {
        const auto y =
            vmaxq_f32(vminq_f32(vaddq_f32(f[i][j], b), vdupq_n_f32(output_max)),
                      vdupq_n_f32(output_min));
        vst1q_f32(outptr + i*output_row_stride + j*output_col_stride, y);
      }
    }
    outptr += 4;
  }
  for (; n_channels >= 2; n_channels -= 2)
  {
    // Matrices used and computed during this transform
    float32x2_t F[4][4], FZ[4][2], f[2][2], b;

    // Read a 4x4 tile in the Winograd domain
    for (auto i = 0u, m = 0u; i < 4; i++)
    {
      for (auto j = 0u; j < 4; j++, m++)
      {
        F[i][j] = vld1_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 2;

    // Compute the matrix F Z
    for (auto i = 0u; i < 4; i++)
    {
      // FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
      FZ[i][0] = vadd_f32(vadd_f32(F[i][0], F[i][1]), F[i][2]);

      // FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
      FZ[i][1] = vsub_f32(vsub_f32(F[i][1], F[i][2]), F[i][3]);
    }

    // Compute the output tile f = ZT F Z
    for (auto j = 0u; j < 2; j++)
    {
      // f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
      f[0][j] = vadd_f32(vadd_f32(FZ[0][j], FZ[1][j]), FZ[2][j]);

      // f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
      f[1][j] = vsub_f32(vsub_f32(FZ[1][j], FZ[2][j]), FZ[3][j]);
    }

    // Load the bias vector
    if (bptr != nullptr)
    {
      b = vld1_f32(bptr);
      bptr += 2;
    }
    else
    {
      b = vdup_n_f32(0.0f);
    }

    // Write out the output tile
    for (auto i = 0u; i < output_tile_rows; i++)
    {
      for (auto j = 0u; j < output_tile_cols; j++)
      {
        const auto y =
            vmax_f32(vmin_f32(vadd_f32(f[i][j], b), vdup_n_f32(output_max)),
                     vdup_n_f32(output_min));
        vst1_f32(outptr + i*output_row_stride + j*output_col_stride, y);
      }
    }
    outptr += 2;
  }
  for (; n_channels; n_channels--)
  {
    // Matrices used and computed during this transform
    float F[4][4], FZ[4][2], f[2][2], b;

    // Read a 4x4 tile in the Winograd domain
    for (auto i = 0u, m = 0u; i < 4; i++)
    {
      for (auto j = 0u; j < 4; j++, m++)
      {
        F[i][j] = *(inptr + m*matrix_stride);
      }
    }
    inptr++;

    // Compute the matrix F Z
    for (auto i = 0u; i < 4; i++)
    {
      FZ[i][0] =  F[i][0] + F[i][1] + F[i][2];
      FZ[i][1] =  F[i][1] - F[i][2] - F[i][3];
    }

    // Compute the output tile f = ZT F Z
    for (auto j = 0u; j < 2; j++)
    {
      f[0][j] =  FZ[0][j] + FZ[1][j] + FZ[2][j];
      f[1][j] =  FZ[1][j] - FZ[2][j] - FZ[3][j];
    }

    // Load the bias
    if (bptr != nullptr)
    {
      b = *(bptr++);
    }
    else
    {
      b = 0.0f;
    }

    // Write out the output tile
    for (auto i = 0u; i < output_tile_rows; i++)
    {
      for (auto j = 0u; j < output_tile_cols; j++)
      {
        const auto y = std::max(std::min(f[i][j] + b, output_max), output_min);
        *(outptr + i*output_row_stride + j*output_col_stride) = y;
      }
    }
    outptr++;
  }
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace arm_conv
