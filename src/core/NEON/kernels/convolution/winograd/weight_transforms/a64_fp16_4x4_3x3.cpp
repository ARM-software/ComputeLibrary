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
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include <cstddef>
#include <arm_neon.h>

namespace arm_conv {
namespace winograd {
namespace weight_transform {

void a64_fp16_4x4_3x3(
    unsigned int n_channels,
    const __fp16* inptr,  // NOTE: Data in HWIO order
    const size_t ld_weight_row,
    const size_t ld_weight_col,
    __fp16* outptr,
    const size_t matrix_stride
)
{
#ifdef __aarch64__
    for (; n_channels >= 8; n_channels -= 8)
    {
      // Matrices used and computed in this kernel
      float16x8_t w[3][3], Ww[6][3], V[6][6];

      // Read weights
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          w[i][j] = vld1q_f16(inptr + i*ld_weight_row + j*ld_weight_col);
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 3; j++)
      {
        // Ww[0][j] =  6*w[0][j];
        Ww[0][j] = vmulq_n_f16(w[0][j], 6.0);

        // Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
        Ww[1][j] = vmulq_n_f16(vaddq_f16(vaddq_f16(w[0][j], w[1][j]), w[2][j]), -4.0);

        // Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
        Ww[2][j] = vmulq_n_f16(vsubq_f16(vsubq_f16(w[1][j], w[0][j]), w[2][j]), 4.0);

        // Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
        Ww[3][j] = vaddq_f16(vaddq_f16(w[0][j], vmulq_f16(w[1][j], vdupq_n_f16(2.0f))), vmulq_f16(w[2][j], vdupq_n_f16(4.0f)));

        // Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
        Ww[4][j] = vaddq_f16(vsubq_f16(w[0][j], vmulq_f16(w[1][j], vdupq_n_f16(2.0f))), vmulq_f16(w[2][j], vdupq_n_f16(4.0f)));

        // Ww[5][j] = 24*w[2][j];
        Ww[5][j] = vmulq_n_f16(w[2][j], 24.0f);
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        const float recip576 = 1.0f / 576.0f;

        // V[i][0] =  6*Ww[i][0];
        V[i][0] = vmulq_n_f16(vmulq_n_f16(Ww[i][0], 6.0), recip576);

        // V[i][1] = -4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2];
        V[i][1] = vmulq_n_f16(vmulq_n_f16(vaddq_f16(vaddq_f16(Ww[i][0], Ww[i][1]), Ww[i][2]), -4.0), recip576);

        // V[i][2] = -4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2];
        V[i][2] = vmulq_n_f16(vmulq_n_f16(vsubq_f16(vsubq_f16(Ww[i][1], Ww[i][0]), Ww[i][2]), 4.0), recip576);

        // V[i][3] =  1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2];
        V[i][3] = vmulq_n_f16(vaddq_f16(vaddq_f16(Ww[i][0], vmulq_f16(Ww[i][1], vdupq_n_f16(2.0f))), vmulq_f16(Ww[i][2], vdupq_n_f16(4.0f))), recip576);

        // V[i][4] =  1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2];
        V[i][4] = vmulq_n_f16(vaddq_f16(vsubq_f16(Ww[i][0], vmulq_f16(Ww[i][1], vdupq_n_f16(2.0f))), vmulq_f16(Ww[i][2], vdupq_n_f16(4.0f))), recip576);

        // V[i][5] = 24*Ww[i][2];
        V[i][5] = vmulq_n_f16(vmulq_n_f16(Ww[i][2], 24.0f), recip576);
      }

      // Store the transformed weights
      for (int i = 0, m = 0; i < 6; i++)
      {
        for (int j = 0; j < 6; j++, m++)
        {
          vst1q_f16(outptr + m*matrix_stride, V[i][j]);
        }
      }
      inptr += 8;
      outptr += 8;
    }
#endif  // __aarch64__
#ifdef __arm_any__
    for (; n_channels >= 4; n_channels -= 4)
    {
      // Matrices used and computed in this kernel
      float16x4_t w[3][3], Ww[6][3], V[6][6];

      // Read weights
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          w[i][j] = vld1_f16(inptr + i*ld_weight_row + j*ld_weight_col);
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 3; j++)
      {
        // Ww[0][j] =  6*w[0][j];
        Ww[0][j] = vmul_n_f16(w[0][j], 6.0);

        // Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
        Ww[1][j] = vmul_n_f16(vadd_f16(vadd_f16(w[0][j], w[1][j]), w[2][j]), -4.0);

        // Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
        Ww[2][j] = vmul_n_f16(vsub_f16(vsub_f16(w[1][j], w[0][j]), w[2][j]), 4.0);

        // Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
        Ww[3][j] = vadd_f16(vadd_f16(w[0][j], vmul_f16(w[1][j], vdup_n_f16(2.0f))), vmul_f16(w[2][j], vdup_n_f16(4.0f)));

        // Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
        Ww[4][j] = vadd_f16(vsub_f16(w[0][j], vmul_f16(w[1][j], vdup_n_f16(2.0f))), vmul_f16(w[2][j], vdup_n_f16(4.0f)));

        // Ww[5][j] = 24*w[2][j];
        Ww[5][j] = vmul_n_f16(w[2][j], 24.0f);
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        const float recip576 = 1.0f / 576.0f;

        // V[i][0] =  6*Ww[i][0];
        V[i][0] = vmul_n_f16(vmul_n_f16(Ww[i][0], 6.0), recip576);

        // V[i][1] = -4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2];
        V[i][1] = vmul_n_f16(vmul_n_f16(vadd_f16(vadd_f16(Ww[i][0], Ww[i][1]), Ww[i][2]), -4.0), recip576);

        // V[i][2] = -4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2];
        V[i][2] = vmul_n_f16(vmul_n_f16(vsub_f16(vsub_f16(Ww[i][1], Ww[i][0]), Ww[i][2]), 4.0), recip576);

        // V[i][3] =  1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2];
        V[i][3] = vmul_n_f16(vadd_f16(vadd_f16(Ww[i][0], vmul_f16(Ww[i][1], vdup_n_f16(2.0f))), vmul_f16(Ww[i][2], vdup_n_f16(4.0f))), recip576);

        // V[i][4] =  1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2];
        V[i][4] = vmul_n_f16(vadd_f16(vsub_f16(Ww[i][0], vmul_f16(Ww[i][1], vdup_n_f16(2.0f))), vmul_f16(Ww[i][2], vdup_n_f16(4.0f))), recip576);

        // V[i][5] = 24*Ww[i][2];
        V[i][5] = vmul_n_f16(vmul_n_f16(Ww[i][2], 24.0f), recip576);
      }

      // Store the transformed weights
      for (int i = 0, m = 0; i < 6; i++)
      {
        for (int j = 0; j < 6; j++, m++)
        {
          vst1_f16(outptr + m*matrix_stride, V[i][j]);
        }
      }
      inptr += 4;
      outptr += 4;
    }
#endif  // __arm_any__
    for (; n_channels; n_channels--)
    {
      // Matrices used and computed in this kernel
      __fp16 w[3][3], Ww[6][3], V[6][6];

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

      inptr++;
      outptr++;
    }
}

}  // namespace weight_transform
}  // namespace winograd
}  // namespace arm_conv

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
