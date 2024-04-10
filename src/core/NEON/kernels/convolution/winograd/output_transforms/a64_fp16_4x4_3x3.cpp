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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#include <algorithm>
#include <arm_neon.h>
#include <cstddef>

namespace arm_conv {
namespace winograd {
namespace output_transform {

void a64_fp16_4x4_3x3(
    unsigned int n_channels,
    const __fp16* inptr,
    const size_t matrix_stride,
    const __fp16* bptr,
    __fp16* const output,
    const size_t output_row_stride,
    const size_t output_col_stride,
    const __fp16 output_min,
    const __fp16 output_max
)
{
    constexpr int output_tile_rows = 4, output_tile_cols = 4;

    // Construct a map to the output cells
    __fp16 *outptrs[output_tile_rows][output_tile_cols];
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
    for (; channels_remaining >= 8; channels_remaining -= 8)
  {
    // Matrices used and computed during this transform
    float16x8_t F[6][6], FZ[6][4], f[4][4], b;

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        F[i][j] = vld1q_f16(inptr + m*matrix_stride);
      }
    }
    inptr += 8;

    // Compute the matrix F Z
    for (int i = 0; i < 6; i++)
    {
      // FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][0] = vaddq_f16(vaddq_f16(vaddq_f16(F[i][0], F[i][1]), vaddq_f16(F[i][2], F[i][3])), F[i][4]);

      // FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
      FZ[i][1] = vaddq_f16(vsubq_f16(F[i][1], F[i][2]), vmulq_f16(vsubq_f16(F[i][3], F[i][4]), vdupq_n_f16(2.0f)));

      // FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
      FZ[i][2] = vaddq_f16(vaddq_f16(F[i][1], F[i][2]), vmulq_f16(vaddq_f16(F[i][3], F[i][4]), vdupq_n_f16(4.0f)));

      // FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
      FZ[i][3] = vaddq_f16(vaddq_f16(vsubq_f16(F[i][1], F[i][2]), vmulq_f16(vsubq_f16(F[i][3], F[i][4]), vdupq_n_f16(8.0f))), F[i][5]);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 4; j++)
    {
      // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[0][j] = vaddq_f16(vaddq_f16(vaddq_f16(FZ[0][j], FZ[1][j]), vaddq_f16(FZ[2][j], FZ[3][j])), FZ[4][j]);

      // f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
      f[1][j] = vaddq_f16(vsubq_f16(FZ[1][j], FZ[2][j]), vmulq_f16(vsubq_f16(FZ[3][j], FZ[4][j]), vdupq_n_f16(2.0f)));

      // f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
      f[2][j] = vaddq_f16(vaddq_f16(FZ[1][j], FZ[2][j]), vmulq_f16(vaddq_f16(FZ[3][j], FZ[4][j]), vdupq_n_f16(4.0f)));

      // f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
      f[3][j] = vaddq_f16(vaddq_f16(vsubq_f16(FZ[1][j], FZ[2][j]), vmulq_f16(vsubq_f16(FZ[3][j], FZ[4][j]), vdupq_n_f16(8.0f))), FZ[5][j]);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1q_f16(bptr);
      bptr += 8;
    }
    else
    {
      b = vdupq_n_f16(0.0f);
    }
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        const auto y =
            vmaxq_f16(vminq_f16(vaddq_f16(f[i][j], b), vdupq_n_f16(output_max)),
                     vdupq_n_f16(output_min));
        vst1q_f16(outptrs[i][j], y);
        outptrs[i][j] += 8;
      }
    }
  }
#endif  // __aarch64__
#ifdef __arm_any__
    for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used and computed during this transform
    float16x4_t F[6][6], FZ[6][4], f[4][4], b;

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        F[i][j] = vld1_f16(inptr + m*matrix_stride);
      }
    }
    inptr += 4;

    // Compute the matrix F Z
    for (int i = 0; i < 6; i++)
    {
      // FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][0] = vadd_f16(vadd_f16(vadd_f16(F[i][0], F[i][1]), vadd_f16(F[i][2], F[i][3])), F[i][4]);

      // FZ[i][1] =  1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4];
      FZ[i][1] = vadd_f16(vsub_f16(F[i][1], F[i][2]), vmul_f16(vsub_f16(F[i][3], F[i][4]), vdup_n_f16(2.0f)));

      // FZ[i][2] =  1*F[i][1] +  1*F[i][2] +  4*F[i][3] +  4*F[i][4];
      FZ[i][2] = vadd_f16(vadd_f16(F[i][1], F[i][2]), vmul_f16(vadd_f16(F[i][3], F[i][4]), vdup_n_f16(4.0f)));

      // FZ[i][3] =  1*F[i][1] + -1*F[i][2] +  8*F[i][3] + -8*F[i][4] +  1*F[i][5];
      FZ[i][3] = vadd_f16(vadd_f16(vsub_f16(F[i][1], F[i][2]), vmul_f16(vsub_f16(F[i][3], F[i][4]), vdup_n_f16(8.0f))), F[i][5]);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 4; j++)
    {
      // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[0][j] = vadd_f16(vadd_f16(vadd_f16(FZ[0][j], FZ[1][j]), vadd_f16(FZ[2][j], FZ[3][j])), FZ[4][j]);

      // f[1][j] =  1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j];
      f[1][j] = vadd_f16(vsub_f16(FZ[1][j], FZ[2][j]), vmul_f16(vsub_f16(FZ[3][j], FZ[4][j]), vdup_n_f16(2.0f)));

      // f[2][j] =  1*FZ[1][j] +  1*FZ[2][j] +  4*FZ[3][j] +  4*FZ[4][j];
      f[2][j] = vadd_f16(vadd_f16(FZ[1][j], FZ[2][j]), vmul_f16(vadd_f16(FZ[3][j], FZ[4][j]), vdup_n_f16(4.0f)));

      // f[3][j] =  1*FZ[1][j] + -1*FZ[2][j] +  8*FZ[3][j] + -8*FZ[4][j] +  1*FZ[5][j];
      f[3][j] = vadd_f16(vadd_f16(vsub_f16(FZ[1][j], FZ[2][j]), vmul_f16(vsub_f16(FZ[3][j], FZ[4][j]), vdup_n_f16(8.0f))), FZ[5][j]);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1_f16(bptr);
      bptr += 4;
    }
    else
    {
      b = vdup_n_f16(0.0f);
    }
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        const auto y =
            vmax_f16(vmin_f16(vadd_f16(f[i][j], b), vdup_n_f16(output_max)),
                     vdup_n_f16(output_min));
        vst1_f16(outptrs[i][j], y);
        outptrs[i][j] += 4;
      }
    }
  }
#endif  // __arm_any__
    for (; channels_remaining; channels_remaining--)
    {
        // Matrices used and computed during this transform
        __fp16 F[6][6], FZ[6][4], f[4][4], b;

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
                const auto y = std::max(std::min<__fp16>(f[i][j] + b, output_max), output_min);
                *(outptrs[i][j]++) = y;
            }
        }
    }
}

} // namespace output_transform
} // namespace winograd
} // namespace arm_conv

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
