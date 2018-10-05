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
void winograd_output_transform_4x4_3x3_fp32_process_tile(
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
  const int pad_bottom = Specialized ? PadBottom : _pad_bottom;
  const int pad_right = Specialized ? PadRight : _pad_right;
  constexpr int TileRows = 4, TileCols = 4;

  const int cells_i = TileRows - pad_bottom;
  const int cells_j = TileCols - pad_right;

  // Construct a map to the output cells
  float *outptrs[TileRows][TileCols];
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
      float32x4_t F[6][6], FZ[6][4], f[4][4], b;

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

        // FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
        FZ[i][1] = vmlaq_n_f32(vsubq_f32(F[i][1], F[i][2]), vsubq_f32(F[i][3], F[i][4]), 2.0f);

        // FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
        FZ[i][2] = vmlaq_n_f32(vaddq_f32(F[i][1], F[i][2]), vaddq_f32(F[i][3], F[i][4]), 4.0f);

        // FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
        FZ[i][3] = vaddq_f32(vmlaq_n_f32(vsubq_f32(F[i][1], F[i][2]), vsubq_f32(F[i][3], F[i][4]), 8.0f), F[i][5]);
      }

      // Compute the output tile f = ZT F Z
      for (int j = 0; j < 4; j++)
      {
        // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
        f[0][j] = vaddq_f32(vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), vaddq_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

        // f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
        f[1][j] = vmlaq_n_f32(vsubq_f32(FZ[1][j], FZ[2][j]), vsubq_f32(FZ[3][j], FZ[4][j]), 2.0f);

        // f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
        f[2][j] = vmlaq_n_f32(vaddq_f32(FZ[1][j], FZ[2][j]), vaddq_f32(FZ[3][j], FZ[4][j]), 4.0f);

        // f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
        f[3][j] = vaddq_f32(vmlaq_n_f32(vsubq_f32(FZ[1][j], FZ[2][j]), vsubq_f32(FZ[3][j], FZ[4][j]), 8.0f), FZ[5][j]);
      }

      // Write out the output tile
      b = vld1q_f32(bptr);
      bptr += 4;
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
      float32x2_t F[6][6], FZ[6][4], f[4][4], b;

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

        // FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
        FZ[i][1] = vmla_n_f32(vsub_f32(F[i][1], F[i][2]), vsub_f32(F[i][3], F[i][4]), 2.0f);

        // FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
        FZ[i][2] = vmla_n_f32(vadd_f32(F[i][1], F[i][2]), vadd_f32(F[i][3], F[i][4]), 4.0f);

        // FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
        FZ[i][3] = vadd_f32(vmla_n_f32(vsub_f32(F[i][1], F[i][2]), vsub_f32(F[i][3], F[i][4]), 8.0f), F[i][5]);
      }

      // Compute the output tile f = ZT F Z
      for (int j = 0; j < 4; j++)
      {
        // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
        f[0][j] = vadd_f32(vadd_f32(vadd_f32(FZ[0][j], FZ[1][j]), vadd_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

        // f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
        f[1][j] = vmla_n_f32(vsub_f32(FZ[1][j], FZ[2][j]), vsub_f32(FZ[3][j], FZ[4][j]), 2.0f);

        // f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
        f[2][j] = vmla_n_f32(vadd_f32(FZ[1][j], FZ[2][j]), vadd_f32(FZ[3][j], FZ[4][j]), 4.0f);

        // f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
        f[3][j] = vadd_f32(vmla_n_f32(vsub_f32(FZ[1][j], FZ[2][j]), vsub_f32(FZ[3][j], FZ[4][j]), 8.0f), FZ[5][j]);
      }

      // Write out the output tile
      b = vld1_f32(bptr);
      bptr += 2;
      for (int i = 0; i < cells_i; i++)
      {
        for (int j = 0; j < cells_j; j++)
        {
          vst1_f32(outptrs[i][j], vadd_f32(f[i][j], b));
          outptrs[i][j] += 2;
        }
      }
    }
#endif
    for (; channels_remaining; channels_remaining--)
    {
      // Matrices used and computed during this transform
      float F[6][6], FZ[6][4], f[4][4], b;

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
        FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
        FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
        FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
      }

      // Compute the output tile f = ZT F Z
      for (int j = 0; j < 4; j++)
      {
        f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
        f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
        f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
        f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
      }

      // Write out the output tile
      b = *(bptr++);
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
      float32x4_t F[6][6], FZ[6][4], f[4][4];

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

        // FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
        FZ[i][1] = vmlaq_n_f32(vsubq_f32(F[i][1], F[i][2]), vsubq_f32(F[i][3], F[i][4]), 2.0f);

        // FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
        FZ[i][2] = vmlaq_n_f32(vaddq_f32(F[i][1], F[i][2]), vaddq_f32(F[i][3], F[i][4]), 4.0f);

        // FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
        FZ[i][3] = vaddq_f32(vmlaq_n_f32(vsubq_f32(F[i][1], F[i][2]), vsubq_f32(F[i][3], F[i][4]), 8.0f), F[i][5]);
      }

      // Compute the output tile f = ZT F Z
      for (int j = 0; j < 4; j++)
      {
        // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
        f[0][j] = vaddq_f32(vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), vaddq_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

        // f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
        f[1][j] = vmlaq_n_f32(vsubq_f32(FZ[1][j], FZ[2][j]), vsubq_f32(FZ[3][j], FZ[4][j]), 2.0f);

        // f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
        f[2][j] = vmlaq_n_f32(vaddq_f32(FZ[1][j], FZ[2][j]), vaddq_f32(FZ[3][j], FZ[4][j]), 4.0f);

        // f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
        f[3][j] = vaddq_f32(vmlaq_n_f32(vsubq_f32(FZ[1][j], FZ[2][j]), vsubq_f32(FZ[3][j], FZ[4][j]), 8.0f), FZ[5][j]);
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
      float32x2_t F[6][6], FZ[6][4], f[4][4];

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

        // FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
        FZ[i][1] = vmla_n_f32(vsub_f32(F[i][1], F[i][2]), vsub_f32(F[i][3], F[i][4]), 2.0f);

        // FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
        FZ[i][2] = vmla_n_f32(vadd_f32(F[i][1], F[i][2]), vadd_f32(F[i][3], F[i][4]), 4.0f);

        // FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
        FZ[i][3] = vadd_f32(vmla_n_f32(vsub_f32(F[i][1], F[i][2]), vsub_f32(F[i][3], F[i][4]), 8.0f), F[i][5]);
      }

      // Compute the output tile f = ZT F Z
      for (int j = 0; j < 4; j++)
      {
        // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
        f[0][j] = vadd_f32(vadd_f32(vadd_f32(FZ[0][j], FZ[1][j]), vadd_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

        // f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
        f[1][j] = vmla_n_f32(vsub_f32(FZ[1][j], FZ[2][j]), vsub_f32(FZ[3][j], FZ[4][j]), 2.0f);

        // f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
        f[2][j] = vmla_n_f32(vadd_f32(FZ[1][j], FZ[2][j]), vadd_f32(FZ[3][j], FZ[4][j]), 4.0f);

        // f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
        f[3][j] = vadd_f32(vmla_n_f32(vsub_f32(FZ[1][j], FZ[2][j]), vsub_f32(FZ[3][j], FZ[4][j]), 8.0f), FZ[5][j]);
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
#endif
    for (; channels_remaining; channels_remaining--)
    {
      // Matrices used and computed during this transform
      float F[6][6], FZ[6][4], f[4][4];

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
        FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
        FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
        FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
      }

      // Compute the output tile f = ZT F Z
      for (int j = 0; j < 4; j++)
      {
        f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
        f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
        f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
        f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
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
using Tiles = OutputTransformImplTiles<3, 3, 6, 6, float>;

template <>
const Tiles::TileFn Tiles::tilefn_generic = winograd_output_transform_4x4_3x3_fp32_process_tile<false>;

template <>
const Tiles::TileFn Tiles::tilefn_unpadded = winograd_output_transform_4x4_3x3_fp32_process_tile<true>;

template <>
const Tiles::TileFn Tiles::tilefn_bottom_padded[n_pad_bottom] = {
  winograd_output_transform_4x4_3x3_fp32_process_tile<true, 1, 0>,
  winograd_output_transform_4x4_3x3_fp32_process_tile<true, 2, 0>,
  winograd_output_transform_4x4_3x3_fp32_process_tile<true, 3, 0>,
};

template <>
const Tiles::TileFn Tiles::tilefn_right_padded[n_pad_right] = {
  winograd_output_transform_4x4_3x3_fp32_process_tile<true, 0, 1>,
  winograd_output_transform_4x4_3x3_fp32_process_tile<true, 0, 2>,
  winograd_output_transform_4x4_3x3_fp32_process_tile<true, 0, 3>,
};

template class OutputTransform<3, 3, 6, 6, float>;
}  // namespace winograd
