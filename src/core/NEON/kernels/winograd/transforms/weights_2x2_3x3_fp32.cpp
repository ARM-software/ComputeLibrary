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

#include "arm.hpp"
#include "winograd_gemm.hpp"
#include "transforms/kernel.hpp"

namespace winograd
{
  template <>
  template <>
  void WinogradGEMM<2, 2, 3, 3>::WeightsTransform<float>::execute(
    const int n_output_channels,
    const int n_input_channels,
    const float* const input,
    float* const output,
    const int matrix_stride,
    const int matrix_row_stride
  )
  {
    constexpr int inner_tile_i = 4;
    constexpr int inner_tile_j = 4;

    // Get pointers to each cell of the weight tensor
    const auto weight_col_stride = n_input_channels * n_output_channels;
    const auto weight_row_stride = 3 * weight_col_stride;
    const float *inptrs[3][3];
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        inptrs[i][j] = input + i*weight_row_stride + j*weight_col_stride;
      }
    }

    // For each input channel
    for (int ic = 0; ic < n_input_channels; ic++)
    {
      float *outptr = output + ic * matrix_row_stride;

      // For each output channel
      int channels_remaining = n_output_channels;
#ifdef __aarch64__
      for (; channels_remaining >= 4; channels_remaining -= 4)
      {
        // Matrices used and computed in this kernel
        float32x4_t w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];

        // Read weights
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < 3; j++)
          {
            w[i][j] = vld1q_f32(inptrs[i][j]);
            inptrs[i][j] += 4;
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
        for (int i = 0; i < inner_tile_i; i++)
        {
          V[i][0] = Ww[i][0];

          // V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
          V[i][1] = vmulq_n_f32(vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

          // V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
          V[i][2] = vmulq_n_f32(vaddq_f32(vsubq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

          V[i][3] = Ww[i][2];
        }

        // Store the transformed weights
        for (int i = 0, m = 0; i < inner_tile_i; i++)
        {
          for (int j = 0; j < inner_tile_j; j++, m++)
          {
            vst1q_f32(outptr + m*matrix_stride, V[i][j]);
          }
        }
        outptr += 4;
      }
#endif  // __aarch64__
#ifdef __arm_any__
      for (; channels_remaining >= 2; channels_remaining -= 2)
      {
        // Matrices used and computed in this kernel
        float32x2_t w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];

        // Read weights
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < 3; j++)
          {
            w[i][j] = vld1_f32(inptrs[i][j]);
            inptrs[i][j] += 2;
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
        for (int i = 0; i < inner_tile_i; i++)
        {
          V[i][0] = Ww[i][0];

          // V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
          V[i][1] = vmul_n_f32(vadd_f32(vadd_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

          // V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
          V[i][2] = vmul_n_f32(vadd_f32(vsub_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), 0.5f);

          V[i][3] = Ww[i][2];
        }

        // Store the transformed weights
        for (int i = 0, m = 0; i < inner_tile_i; i++)
        {
          for (int j = 0; j < inner_tile_j; j++, m++)
          {
            vst1_f32(outptr + m*matrix_stride, V[i][j]);
          }
        }
        outptr += 2;
      }
#endif  // __arm_any__
      for (; channels_remaining; channels_remaining--)
      {
        // Matrices used and computed in this kernel
        float w[3][3], Ww[inner_tile_i][3], V[inner_tile_i][inner_tile_j];

        // Read weights
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < 3; j++)
          {
            w[i][j] = *(inptrs[i][j]++);
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
        for (int i = 0; i < inner_tile_i; i++)
        {
          V[i][0] = Ww[i][0];
          V[i][1] = 0.5*(Ww[i][0] + Ww[i][1] + Ww[i][2]);
          V[i][2] = 0.5*(Ww[i][0] - Ww[i][1] + Ww[i][2]);
          V[i][3] = Ww[i][2];
        }

        // Store the transformed weights
        for (int i = 0, m = 0; i < inner_tile_i; i++)
        {
          for (int j = 0; j < inner_tile_j; j++, m++)
          {
            *(outptr + m*matrix_stride) = V[i][j];
          }
        }
        outptr++;
      }
    }
  }

  template <>
  template <>
  int WinogradGEMM<2, 2, 3, 3>::WeightsTransform<float>::ops_performed(const KernelShape &shape)
  {
    const int channel_prod = shape.n_input_channels * shape.n_output_channels;
    return 2 * 18 * channel_prod;
  }

  template struct WinogradGEMM<2, 2, 3, 3>::WeightsTransform<float>;
}  // namespace winograd
