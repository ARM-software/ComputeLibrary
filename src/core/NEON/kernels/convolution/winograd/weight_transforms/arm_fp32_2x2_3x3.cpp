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

#include <cstddef>
#include <arm_neon.h>

namespace arm_conv {
namespace winograd {
namespace weight_transform {

void arm_fp32_2x2_3x3(
  unsigned int n_channels,
  const float *inptr, size_t ld_weight_row, size_t ld_weight_col,
  float *outptr, size_t matrix_stride
)
{
  constexpr auto inner_tile_i = 4u;
  constexpr auto inner_tile_j = 4u;

#ifdef __aarch64__
  // For each output channel
  for (; n_channels >= 4u; n_channels -= 4)
  {
    // Matrices used and computed in this kernel
    float32x4_t w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];

    // Read weights
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        w[i][j] = vld1q_f32(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 3; j++)
    {
      Ww[0][j] = w[0][j];

      // Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j]);
      Ww[1][j] = vmulq_n_f32(vaddq_f32(vaddq_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);

      // Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j]);
      Ww[2][j] = vmulq_n_f32(vaddq_f32(vsubq_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);

      Ww[3][j] = w[2][j];
    }

    // Compute V = W w WT
    for (auto i = 0u; i < inner_tile_i; i++)
    {
      V[i][0] = Ww[i][0];

      // V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
      V[i][1] = vmulq_n_f32(vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

      // V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
      V[i][2] = vmulq_n_f32(vaddq_f32(vsubq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

      V[i][3] = Ww[i][2];
    }

    // Store the transformed weights
    for (auto i = 0u, m = 0u; i < inner_tile_i; i++)
    {
      for (auto j = 0u; j < inner_tile_j; j++, m++)
      {
        vst1q_f32(outptr + m*matrix_stride, V[i][j]);
      }
    }

    inptr += 4;
    outptr += 4;
  }
#endif // __aarch64__
  for (; n_channels >= 2u; n_channels -= 2)
  {
    // Matrices used and computed in this kernel
    float32x2_t w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];

    // Read weights
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        w[i][j] = vld1_f32(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 3; j++)
    {
      Ww[0][j] = w[0][j];

      // Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j]);
      Ww[1][j] = vmul_n_f32(vadd_f32(vadd_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);

      // Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j]);
      Ww[2][j] = vmul_n_f32(vadd_f32(vsub_f32(w[0][j], w[1][j]), w[2][j]), 0.5f);

      Ww[3][j] = w[2][j];
    }

    // Compute V = W w WT
    for (auto i = 0u; i < inner_tile_i; i++)
    {
      V[i][0] = Ww[i][0];

      // V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
      V[i][1] = vmul_n_f32(vadd_f32(vadd_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

      // V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
      V[i][2] = vmul_n_f32(vadd_f32(vsub_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

      V[i][3] = Ww[i][2];
    }

    // Store the transformed weights
    for (auto i = 0u, m = 0u; i < inner_tile_i; i++)
    {
      for (auto j = 0u; j < inner_tile_j; j++, m++)
      {
        vst1_f32(outptr + m*matrix_stride, V[i][j]);
      }
    }

    inptr += 2;
    outptr += 2;
  }
  for (; n_channels; n_channels--)
  {
    // Matrices used and computed in this kernel
    float w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];

    // Read weights
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        w[i][j] = *(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 3; j++)
    {
      Ww[0][j] = w[0][j];
      Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j]);
      Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j]);
      Ww[3][j] = w[2][j];
    }

    // Compute V = W w WT
    for (auto i = 0u; i < inner_tile_i; i++)
    {
      V[i][0] = Ww[i][0];
      V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
      V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
      V[i][3] = Ww[i][2];
    }

    // Store the transformed weights
    for (auto i = 0u, m = 0u; i < inner_tile_i; i++)
    {
      for (auto j = 0u; j < inner_tile_j; j++, m++)
      {
        *(outptr + m*matrix_stride) = V[i][j];
      }
    }

    inptr++;
    outptr++;
  }
}

}  // namespace weight_transform
}  // namespace winograd
}  // namespace arm_conv
