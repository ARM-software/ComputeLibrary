/*
 * Copyright (c) 2017-2019 Arm Limited.
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

#include "output.hpp"
#include "arm.hpp"

namespace winograd
{

template <>
void OutputTransform<5, 5, 6, 6, float, float, WinogradRoots::Integers>::transform_tile(
  const int n_channels,
  const float* inptr,
  const int matrix_stride,
  const float* bptr,
  float* const output,
  const int output_row_stride,
  const int output_col_stride,
  const float output_min,
  const float output_max
)
{
  // Construct a map to the output cells
  float *outptrs[output_tile_rows][output_tile_cols];
  for (int i = 0; i < output_tile_rows; i++)
  {
    for (int j = 0; j < output_tile_cols; j++)
    {
      outptrs[i][j] = output + i*output_row_stride + j*output_col_stride;
    }
  }

  // For each channel of the output
  int channels_remaining = n_channels;
#ifdef __aarch64__
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used and computed during this transform
    float32x4_t F[6][6], FZ[6][2], f[2][2], b;

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        F[i][j] = vld1q_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 4;

    // Compute the matrix F Z
    for (int i = 0; i < 6; i++)
    {
      // FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][0] = vaddq_f32(vaddq_f32(vaddq_f32(F[i][0], F[i][1]), vaddq_f32(F[i][2], F[i][3])), F[i][4]);

      // FZ[i][1] =               1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4] +  1*F[i][5];
      FZ[i][1] = vaddq_f32(vmlaq_n_f32(vsubq_f32(F[i][1], F[i][2]), vsubq_f32(F[i][3], F[i][4]), 2.0f), F[i][5]);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 2; j++)
    {
      // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[0][j] = vaddq_f32(vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), vaddq_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

      // f[1][j] =               1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j] +  1*FZ[5][j];
      f[1][j] = vaddq_f32(vmlaq_n_f32(vsubq_f32(FZ[1][j], FZ[2][j]), vsubq_f32(FZ[3][j], FZ[4][j]), 2.0f), FZ[5][j]);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1q_f32(bptr);
      bptr += 4;
    }
    else
    {
      b = vdupq_n_f32(0.0f);
    }
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        const auto y =
            vmaxq_f32(vminq_f32(vaddq_f32(f[i][j], b), vdupq_n_f32(output_max)),
                      vdupq_n_f32(output_min));
        vst1q_f32(outptrs[i][j], y);
        outptrs[i][j] += 4;
      }
    }
  }
#endif  // __aarch64__
#ifdef __arm_any__
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    // Matrices used and computed during this transform
    float32x2_t F[6][6], FZ[6][2], f[2][2], b;

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        F[i][j] = vld1_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 2;

    // Compute the matrix F Z
    for (int i = 0; i < 6; i++)
    {
      // FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][0] = vadd_f32(vadd_f32(vadd_f32(F[i][0], F[i][1]), vadd_f32(F[i][2], F[i][3])), F[i][4]);

      // FZ[i][1] =               1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4] +  1*F[i][5];
      FZ[i][1] = vadd_f32(vmla_n_f32(vsub_f32(F[i][1], F[i][2]), vsub_f32(F[i][3], F[i][4]), 2.0f), F[i][5]);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 2; j++)
    {
      // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[0][j] = vadd_f32(vadd_f32(vadd_f32(FZ[0][j], FZ[1][j]), vadd_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

      // f[1][j] =               1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j] +  1*FZ[5][j];
      f[1][j] = vadd_f32(vmla_n_f32(vsub_f32(FZ[1][j], FZ[2][j]), vsub_f32(FZ[3][j], FZ[4][j]), 2.0f), FZ[5][j]);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1_f32(bptr);
      bptr += 2;
    }
    else
    {
      b = vdup_n_f32(0.0f);
    }
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        const auto y =
            vmax_f32(vmin_f32(vadd_f32(f[i][j], b), vdup_n_f32(output_max)),
                     vdup_n_f32(output_min));
        vst1_f32(outptrs[i][j], y);
        outptrs[i][j] += 2;
      }
    }
  }
#endif  // __arm_any__
  for (; channels_remaining; channels_remaining--)
  {
    // Matrices used and computed during this transform
    float F[6][6], FZ[6][2], f[2][2], b;

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        F[i][j] = *(inptr + m*matrix_stride);
      }
    }
    inptr++;

    // Compute the matrix F Z
    for (int i = 0; i < 6; i++)
    {
      FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][1] =               1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4] +  1*F[i][5];
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 2; j++)
    {
      f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[1][j] =                1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j] +  1*FZ[5][j];
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = *(bptr++);
    }
    else
    {
      b = 0.0f;
    }
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        const auto y = std::max(std::min(f[i][j] + b, output_max), output_min);
        *(outptrs[i][j]++) = y;
      }
    }
  }
}

template class OutputTransform<5, 5, 6, 6, float, float, WinogradRoots::Integers>;

}  // namespace
