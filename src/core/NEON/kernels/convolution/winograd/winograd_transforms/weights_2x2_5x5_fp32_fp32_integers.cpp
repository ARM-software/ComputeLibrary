/*
 * Copyright (c) 2019 ARM Limited.
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
#include "kernel.hpp"

namespace winograd
{

template <>
void WeightTransform<5, 5, 6, 6, float, float, WinogradRoots::Integers>::execute(
  const int n_output_channels,
  const int n_input_channels,
  const float* const input,
  float* const output,
  const int matrix_stride,
  const int matrix_row_stride
)
{
  // Get pointers to each cell of the weight tensor
  const auto weight_col_stride = n_input_channels * n_output_channels;
  const auto weight_row_stride = 5 * weight_col_stride;
  const float *inptrs[5][5];
  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 5; j++)
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
      float32x4_t w[5][5], Ww[6][5], V[6][6];

      // Read weights
      for (int i = 0; i < 5; i++)
      {
        for (int j = 0; j < 5; j++)
        {
          w[i][j] = vld1q_f32(inptrs[i][j]);
          inptrs[i][j] += 4;
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 5; j++)
      {
        // Ww[0][j] = w[0][j]/4.0f;
        Ww[0][j] = vmulq_n_f32(w[0][j], 1.0f/4.0f);

        // Ww[1][j] = -( w[0][j] + w[1][j] + w[2][j] + w[3][j] + w[4][j])/6.0f;
        Ww[1][j] = vmulq_n_f32(
          vaddq_f32(
            vaddq_f32(
              vaddq_f32(w[1][j], w[0][j]),
              vaddq_f32(w[3][j], w[2][j])
            ),
            w[4][j]
          ),
          -1.0f/6.0f
        );

        // Ww[2][j] = +(-w[0][j] + w[1][j] - w[2][j] + w[3][j] - w[4][j])/6.0f;
        // Ww[2][j] = ((w[1][j] - w[0][j]) + (w[3][j] - w[2][j]) - w[4][j])/6.0f;
        Ww[2][j] = vmulq_n_f32(
          vsubq_f32(
            vaddq_f32(
              vsubq_f32(w[1][j], w[0][j]),
              vsubq_f32(w[3][j], w[2][j])
            ),
            w[4][j]
          ),
          1.0f/6.0f
        );

        // Ww[3][j] = (w[0][j]/8.0f + w[1][j]/4.0f + w[2][j]/2.0f + w[3][j] + 2*w[4][j])/3.0f;
        Ww[3][j] = vmulq_n_f32(
          vmlaq_n_f32(
            vaddq_f32(
              vaddq_f32(vmulq_n_f32(w[0][j], 1.0f/8.0f), vmulq_n_f32(w[1][j], 1.0f/4.0f)),
              vaddq_f32(vmulq_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
            ),
            w[4][j], 2.0f
          ),
          1.0f/3.0f
        );

        // Ww[4][j] = (w[0][j]/8.0f - w[1][j]/4.0f + w[2][j]/2.0f - w[3][j] + 2*w[4][j])/3.0f;
        Ww[4][j] = vmulq_n_f32(
          vmlaq_n_f32(
            vaddq_f32(
              vsubq_f32(vmulq_n_f32(w[0][j], 1.0f/8.0f), vmulq_n_f32(w[1][j], 1.0f/4.0f)),
              vsubq_f32(vmulq_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
            ),
            w[4][j], 2.0f
          ),
          1.0f/3.0f
        );

        // Ww[5][j] = w[4][j];
        Ww[5][j] = w[4][j];
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        // V[i][0] = Ww[i][0]/4.0f;
        V[i][0] = vmulq_n_f32(Ww[i][0], 1.0f/4.0f);

        // V[i][1] = -( Ww[i][0] + Ww[i][1] + Ww[i][2] + Ww[i][3] + Ww[i][4])/6.0f;
        V[i][1] = vmulq_n_f32(
          vaddq_f32(
            vaddq_f32(
              vaddq_f32(Ww[i][1], Ww[i][0]),
              vaddq_f32(Ww[i][3], Ww[i][2])
            ),
            Ww[i][4]
          ),
          -1.0f/6.0f
        );

        // V[i][2] = +(-Ww[i][0] + Ww[i][1] - Ww[i][2] + Ww[i][3] - Ww[i][4])/6.0f;
        // V[i][2] = ((Ww[i][1] - Ww[i][0]) + (Ww[i][3] - Ww[i][2]) - Ww[i][4])/6.0f;
        V[i][2] = vmulq_n_f32(
          vsubq_f32(
            vaddq_f32(
              vsubq_f32(Ww[i][1], Ww[i][0]),
              vsubq_f32(Ww[i][3], Ww[i][2])
            ),
            Ww[i][4]
          ),
          1.0f/6.0f
        );

        // V[i][3] = (Ww[i][0]/8.0f + Ww[i][1]/4.0f + Ww[i][2]/2.0f + Ww[i][3] + 2*Ww[i][4])/3.0f;
        V[i][3] = vmulq_n_f32(
          vmlaq_n_f32(
            vaddq_f32(
              vaddq_f32(vmulq_n_f32(Ww[i][0], 1.0f/8.0f), vmulq_n_f32(Ww[i][1], 1.0f/4.0f)),
              vaddq_f32(vmulq_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
            ),
            Ww[i][4], 2.0f
          ),
          1.0f/3.0f
        );

        // V[i][4] = (Ww[i][0]/8.0f - Ww[i][1]/4.0f + Ww[i][2]/2.0f - Ww[i][3] + 2*Ww[i][4])/3.0f;
        V[i][4] = vmulq_n_f32(
          vmlaq_n_f32(
            vaddq_f32(
              vsubq_f32(vmulq_n_f32(Ww[i][0], 1.0f/8.0f), vmulq_n_f32(Ww[i][1], 1.0f/4.0f)),
              vsubq_f32(vmulq_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
            ),
            Ww[i][4], 2.0f
          ),
          1.0f/3.0f
        );

        // V[i][5] = Ww[i][4];
        V[i][5] = Ww[i][4];
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
      float32x2_t w[5][5], Ww[6][5], V[6][6];

      // Read weights
      for (int i = 0; i < 5; i++)
      {
        for (int j = 0; j < 5; j++)
        {
          w[i][j] = vld1_f32(inptrs[i][j]);
          inptrs[i][j] += 2;
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 5; j++)
      {
        // Ww[0][j] = w[0][j]/4.0f;
        Ww[0][j] = vmul_n_f32(w[0][j], 1.0f/4.0f);

        // Ww[1][j] = -( w[0][j] + w[1][j] + w[2][j] + w[3][j] + w[4][j])/6.0f;
        Ww[1][j] = vmul_n_f32(
          vadd_f32(
            vadd_f32(
              vadd_f32(w[1][j], w[0][j]),
              vadd_f32(w[3][j], w[2][j])
            ),
            w[4][j]
          ),
          -1.0f/6.0f
        );

        // Ww[2][j] = +(-w[0][j] + w[1][j] - w[2][j] + w[3][j] - w[4][j])/6.0f;
        // Ww[2][j] = ((w[1][j] - w[0][j]) + (w[3][j] - w[2][j]) - w[4][j])/6.0f;
        Ww[2][j] = vmul_n_f32(
          vsub_f32(
            vadd_f32(
              vsub_f32(w[1][j], w[0][j]),
              vsub_f32(w[3][j], w[2][j])
            ),
            w[4][j]
          ),
          1.0f/6.0f
        );

        // Ww[3][j] = (w[0][j]/8.0f + w[1][j]/4.0f + w[2][j]/2.0f + w[3][j] + 2*w[4][j])/3.0f;
        Ww[3][j] = vmul_n_f32(
          vmla_n_f32(
            vadd_f32(
              vadd_f32(vmul_n_f32(w[0][j], 1.0f/8.0f), vmul_n_f32(w[1][j], 1.0f/4.0f)),
              vadd_f32(vmul_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
            ),
            w[4][j], 2.0f
          ),
          1.0f/3.0f
        );

        // Ww[4][j] = (w[0][j]/8.0f - w[1][j]/4.0f + w[2][j]/2.0f - w[3][j] + 2*w[4][j])/3.0f;
        Ww[4][j] = vmul_n_f32(
          vmla_n_f32(
            vadd_f32(
              vsub_f32(vmul_n_f32(w[0][j], 1.0f/8.0f), vmul_n_f32(w[1][j], 1.0f/4.0f)),
              vsub_f32(vmul_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
            ),
            w[4][j], 2.0f
          ),
          1.0f/3.0f
        );

        // Ww[5][j] = w[4][j];
        Ww[5][j] = w[4][j];
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        // V[i][0] = Ww[i][0]/4.0f;
        V[i][0] = vmul_n_f32(Ww[i][0], 1.0f/4.0f);

        // V[i][1] = -( Ww[i][0] + Ww[i][1] + Ww[i][2] + Ww[i][3] + Ww[i][4])/6.0f;
        V[i][1] = vmul_n_f32(
          vadd_f32(
            vadd_f32(
              vadd_f32(Ww[i][1], Ww[i][0]),
              vadd_f32(Ww[i][3], Ww[i][2])
            ),
            Ww[i][4]
          ),
          -1.0f/6.0f
        );

        // V[i][2] = +(-Ww[i][0] + Ww[i][1] - Ww[i][2] + Ww[i][3] - Ww[i][4])/6.0f;
        // V[i][2] = ((Ww[i][1] - Ww[i][0]) + (Ww[i][3] - Ww[i][2]) - Ww[i][4])/6.0f;
        V[i][2] = vmul_n_f32(
          vsub_f32(
            vadd_f32(
              vsub_f32(Ww[i][1], Ww[i][0]),
              vsub_f32(Ww[i][3], Ww[i][2])
            ),
            Ww[i][4]
          ),
          1.0f/6.0f
        );

        // V[i][3] = (Ww[i][0]/8.0f + Ww[i][1]/4.0f + Ww[i][2]/2.0f + Ww[i][3] + 2*Ww[i][4])/3.0f;
        V[i][3] = vmul_n_f32(
          vmla_n_f32(
            vadd_f32(
              vadd_f32(vmul_n_f32(Ww[i][0], 1.0f/8.0f), vmul_n_f32(Ww[i][1], 1.0f/4.0f)),
              vadd_f32(vmul_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
            ),
            Ww[i][4], 2.0f
          ),
          1.0f/3.0f
        );

        // V[i][4] = (Ww[i][0]/8.0f - Ww[i][1]/4.0f + Ww[i][2]/2.0f - Ww[i][3] + 2*Ww[i][4])/3.0f;
        V[i][4] = vmul_n_f32(
          vmla_n_f32(
            vadd_f32(
              vsub_f32(vmul_n_f32(Ww[i][0], 1.0f/8.0f), vmul_n_f32(Ww[i][1], 1.0f/4.0f)),
              vsub_f32(vmul_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
            ),
            Ww[i][4], 2.0f
          ),
          1.0f/3.0f
        );

        // V[i][5] = Ww[i][4];
        V[i][5] = Ww[i][4];
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
      float w[5][5], Ww[6][5], V[6][6];

      // Read weights
      for (int i = 0; i < 5; i++)
      {
        for (int j = 0; j < 5; j++)
        {
          w[i][j] = *(inptrs[i][j]++);
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 5; j++)
      {
        Ww[0][j] = w[0][j]/4.0f;
        Ww[1][j] = -( w[0][j] + w[1][j] + w[2][j] + w[3][j] + w[4][j])/6.0f;
        Ww[2][j] = +(-w[0][j] + w[1][j] - w[2][j] + w[3][j] - w[4][j])/6.0f;
        Ww[3][j] = (w[0][j]/8.0f + w[1][j]/4.0f + w[2][j]/2.0f + w[3][j] + 2*w[4][j])/3.0f;
        Ww[4][j] = (w[0][j]/8.0f - w[1][j]/4.0f + w[2][j]/2.0f - w[3][j] + 2*w[4][j])/3.0f;
        Ww[5][j] = w[4][j];
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        V[i][0] = Ww[i][0]/4.0f;
        V[i][1] = -( Ww[i][0] + Ww[i][1] + Ww[i][2] + Ww[i][3] + Ww[i][4])/6.0f;
        V[i][2] = +(-Ww[i][0] + Ww[i][1] - Ww[i][2] + Ww[i][3] - Ww[i][4])/6.0f;
        V[i][3] = (Ww[i][0]/8.0f + Ww[i][1]/4.0f + Ww[i][2]/2.0f + Ww[i][3] + 2*Ww[i][4])/3.0f;
        V[i][4] = (Ww[i][0]/8.0f - Ww[i][1]/4.0f + Ww[i][2]/2.0f - Ww[i][3] + 2*Ww[i][4])/3.0f;
        V[i][5] = Ww[i][4];
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

template class WeightTransform<5, 5, 6, 6, float, float, WinogradRoots::Integers>;

}  // namespace winograd
