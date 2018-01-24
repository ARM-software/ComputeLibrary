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
  /* Float implementation for kernel transform F(4x4, 3x3) */
  template <>
  template <>
  void WinogradGEMM<4, 4, 3, 3>::WeightsTransform<float>::execute(
    const int n_output_channels,
    const int n_input_channels,
    const float* const input,  // NOTE: Data in HWIO order
    float* const output,
    const int matrix_stride,
    const int matrix_row_stride
  )
  {
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
        float32x4_t w[3][3], Ww[6][3], V[6][6];

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
          // Ww[0][j] =  6*w[0][j];
          Ww[0][j] = vmulq_n_f32(w[0][j], 6.0);

          // Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
          Ww[1][j] = vmulq_n_f32(vaddq_f32(vaddq_f32(w[0][j], w[1][j]), w[2][j]), -4.0);

          // Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
          Ww[2][j] = vmulq_n_f32(vsubq_f32(vsubq_f32(w[1][j], w[0][j]), w[2][j]), 4.0);

          // Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
          Ww[3][j] = vmlaq_n_f32(vmlaq_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

          // Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
          Ww[4][j] = vmlaq_n_f32(vmlsq_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

          // Ww[5][j] = 24*w[2][j];
          Ww[5][j] = vmulq_n_f32(w[2][j], 24.0f);
        }

        // Compute V = W w WT
        for (int i = 0; i < 6; i++)
        {
          const float recip576 = 1.0f / 576.0f;

          // V[i][0] =  6*Ww[i][0];
          V[i][0] = vmulq_n_f32(vmulq_n_f32(Ww[i][0], 6.0), recip576);

          // V[i][1] = -4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2];
          V[i][1] = vmulq_n_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), -4.0), recip576);

          // V[i][2] = -4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2];
          V[i][2] = vmulq_n_f32(vmulq_n_f32(vsubq_f32(vsubq_f32(Ww[i][1], Ww[i][0]), Ww[i][2]), 4.0), recip576);

          // V[i][3] =  1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2];
          V[i][3] = vmulq_n_f32(vmlaq_n_f32(vmlaq_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

          // V[i][4] =  1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2];
          V[i][4] = vmulq_n_f32(vmlaq_n_f32(vmlsq_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

          // V[i][5] = 24*Ww[i][2];
          V[i][5] = vmulq_n_f32(vmulq_n_f32(Ww[i][2], 24.0f), recip576);
        }

        // Store the transformed weights
        for (int i = 0, m = 0; i < 6; i++)
        {
          for (int j = 0; j < 6; j++, m++)
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
        float32x2_t w[3][3], Ww[6][3], V[6][6];

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
          // Ww[0][j] =  6*w[0][j];
          Ww[0][j] = vmul_n_f32(w[0][j], 6.0);

          // Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
          Ww[1][j] = vmul_n_f32(vadd_f32(vadd_f32(w[0][j], w[1][j]), w[2][j]), -4.0);

          // Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
          Ww[2][j] = vmul_n_f32(vsub_f32(vsub_f32(w[1][j], w[0][j]), w[2][j]), 4.0);

          // Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
          Ww[3][j] = vmla_n_f32(vmla_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

          // Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
          Ww[4][j] = vmla_n_f32(vmls_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

          // Ww[5][j] = 24*w[2][j];
          Ww[5][j] = vmul_n_f32(w[2][j], 24.0f);
        }

        // Compute V = W w WT
        for (int i = 0; i < 6; i++)
        {
          const float recip576 = 1.0f / 576.0f;

          // V[i][0] =  6*Ww[i][0];
          V[i][0] = vmul_n_f32(vmul_n_f32(Ww[i][0], 6.0), recip576);

          // V[i][1] = -4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2];
          V[i][1] = vmul_n_f32(vmul_n_f32(vadd_f32(vadd_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), -4.0), recip576);

          // V[i][2] = -4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2];
          V[i][2] = vmul_n_f32(vmul_n_f32(vsub_f32(vsub_f32(Ww[i][1], Ww[i][0]), Ww[i][2]), 4.0), recip576);

          // V[i][3] =  1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2];
          V[i][3] = vmul_n_f32(vmla_n_f32(vmla_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

          // V[i][4] =  1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2];
          V[i][4] = vmul_n_f32(vmla_n_f32(vmls_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

          // V[i][5] = 24*Ww[i][2];
          V[i][5] = vmul_n_f32(vmul_n_f32(Ww[i][2], 24.0f), recip576);
        }

        // Store the transformed weights
        for (int i = 0, m = 0; i < 6; i++)
        {
          for (int j = 0; j < 6; j++, m++)
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
        float w[3][3], Ww[6][3], V[6][6];

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
          Ww[0][j] =  6*w[0][j];
          Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
          Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
          Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
          Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
          Ww[5][j] = 24*w[2][j];
        }

        // Compute V = W w WT
        for (int i = 0; i < 6; i++)
        {
          V[i][0] = ( 6*Ww[i][0]) / 576.0;
          V[i][1] = (-4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2]) / 576.0;
          V[i][2] = (-4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2]) / 576.0;
          V[i][3] = ( 1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2]) / 576.0;
          V[i][4] = ( 1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2]) / 576.0;
          V[i][5] = (24*Ww[i][2]) / 576.0;
        }

        // Store the transformed weights
        for (int i = 0, m = 0; i < 6; i++)
        {
          for (int j = 0; j < 6; j++, m++)
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
  int WinogradGEMM<4, 4, 3, 3>::WeightsTransform<float>::ops_performed(const KernelShape &shape)
  {
    const int channel_prod = shape.n_input_channels * shape.n_output_channels;
    return 9 * 16 * channel_prod;
  }

  template struct WinogradGEMM<4, 4, 3, 3>::WeightsTransform<float>;
}
