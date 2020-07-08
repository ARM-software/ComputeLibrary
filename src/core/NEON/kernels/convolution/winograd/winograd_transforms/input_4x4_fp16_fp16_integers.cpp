/*
 * Copyright (c) 2020 Arm Limited.
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

#include "input.hpp"
#include "arm.hpp"

namespace winograd
{

template <>
void InputTransform<4, 4, __fp16, __fp16, WinogradRoots::Integers>::transform_tile(
    const int n_channels,
    const __fp16* const input_base,
    const int input_row_stride,
    const int input_col_stride,
    __fp16* outptr,
    const int matrix_stride
)
{
    constexpr int inner_tile_rows = 4, inner_tile_cols = 4;

    // Get pointers into the input tile
    const __fp16 *x_ptrs[inner_tile_rows][inner_tile_cols];
    for (int i = 0, xi = 0; i < inner_tile_rows; i++, xi++)
    {
        // Get a pointer into the row
        const __fp16* const row_ptr = input_base + xi*input_row_stride;

        for (int j = 0, xj = 0; j < inner_tile_cols; j++, xj++)
        {
            x_ptrs[i][j] = row_ptr + xj*input_col_stride;
        }
    }

    // Matrices used/computed in this kernel.
    __fp16 x[inner_tile_rows][inner_tile_cols];
    __fp16 XTx[inner_tile_rows][inner_tile_cols];
    __fp16 U[inner_tile_rows][inner_tile_cols];

    for (int i = 0; i < inner_tile_rows; i++)
    {
        for (int j = 0; j < inner_tile_cols; j++)
        {
            x[i][j] = XTx[i][j] = 0.0f;
        }
    }

    // Perform the Winograd input transformation for each channel in the input
    // tensor.
    int channels_remaining = n_channels;
#ifdef __aarch64__
    for (; channels_remaining >= 8; channels_remaining -= 8)
  {
    // Matrices used/computed in this kernel.
    float16x8_t x[inner_tile_rows][inner_tile_cols];
    float16x8_t XTx[inner_tile_rows][inner_tile_cols];
    float16x8_t U[inner_tile_rows][inner_tile_cols];

    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vdupq_n_f16(0.0f);
        XTx[i][j] = vdupq_n_f16(0.0f);
      }
    }

    // Load x
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vld1q_f16(x_ptrs[i][j]);
        x_ptrs[i][j] += 8;
      }
    }

    // Compute XT . x
    for (int j = 0; j < inner_tile_cols; j++)
    {
      // XTx[0][j] = x[0][j] - x[2][j];
      XTx[0][j] = vsubq_f16(x[0][j], x[2][j]);

      // XTx[1][j] = x[1][j] + x[2][j];
      XTx[1][j] = vaddq_f16(x[1][j], x[2][j]);

      // XTx[2][j] = x[2][j] - x[1][j];
      XTx[2][j] = vsubq_f16(x[2][j], x[1][j]);

      // XTx[3][j] = x[1][j] - x[3][j];
      XTx[3][j] = vsubq_f16(x[1][j], x[3][j]);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      // U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][0] = vsubq_f16(XTx[i][0], XTx[i][2]);

      // U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][1] = vaddq_f16(XTx[i][1], XTx[i][2]);

      // U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][2] = vsubq_f16(XTx[i][2], XTx[i][1]);

      // U[i][3] = XTx[i][1] - XTx[i][3];
      U[i][3] = vsubq_f16(XTx[i][1], XTx[i][3]);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++, m++)
      {
        vst1q_f16(outptr + m*matrix_stride, U[i][j]);
      }
    }
    outptr += 8;
  }
#endif  // __aarch64__
#ifdef __arm_any__
    for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used/computed in this kernel.
    float16x4_t x[inner_tile_rows][inner_tile_cols];
    float16x4_t XTx[inner_tile_rows][inner_tile_cols];
    float16x4_t U[inner_tile_rows][inner_tile_cols];

    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vdup_n_f16(0.0f);
        XTx[i][j] = vdup_n_f16(0.0f);
      }
    }

    // Load x
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vld1_f16(x_ptrs[i][j]);
        x_ptrs[i][j] += 4;
      }
    }

    // Compute XT . x
    for (int j = 0; j < inner_tile_cols; j++)
    {
      // XTx[0][j] = x[0][j] - x[2][j];
      XTx[0][j] = vsub_f16(x[0][j], x[2][j]);

      // XTx[1][j] = x[1][j] + x[2][j];
      XTx[1][j] = vadd_f16(x[1][j], x[2][j]);

      // XTx[2][j] = x[2][j] - x[1][j];
      XTx[2][j] = vsub_f16(x[2][j], x[1][j]);

      // XTx[3][j] = x[1][j] - x[3][j];
      XTx[3][j] = vsub_f16(x[1][j], x[3][j]);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      // U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][0] = vsub_f16(XTx[i][0], XTx[i][2]);

      // U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][1] = vadd_f16(XTx[i][1], XTx[i][2]);

      // U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][2] = vsub_f16(XTx[i][2], XTx[i][1]);

      // U[i][3] = XTx[i][1] - XTx[i][3];
      U[i][3] = vsub_f16(XTx[i][1], XTx[i][3]);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++, m++)
      {
        vst1_f16(outptr + m*matrix_stride, U[i][j]);
      }
    }
    outptr += 4;
  }
#endif  // __arm_any__
    for (; channels_remaining; channels_remaining--)
    {
        // Load x
        for (int i = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++)
            {
                x[i][j] = *(x_ptrs[i][j]++);
            }
        }

        // Compute XT . x
        for (int j = 0; j < inner_tile_cols; j++)
        {
            XTx[0][j] = x[0][j] - x[2][j];
            XTx[1][j] = x[1][j] + x[2][j];
            XTx[2][j] = x[2][j] - x[1][j];
            XTx[3][j] = x[1][j] - x[3][j];
        }

        // Compute U = XT . x . X
        for (int i = 0; i < inner_tile_rows; i++)
        {
            U[i][0] = XTx[i][0] - XTx[i][2];
            U[i][1] = XTx[i][1] + XTx[i][2];
            U[i][2] = XTx[i][2] - XTx[i][1];
            U[i][3] = XTx[i][1] - XTx[i][3];
        }

        // Store the transformed matrix
        for (int i = 0, m = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++, m++)
            {
                *(outptr + m*matrix_stride) = U[i][j];
            }
        }
        outptr++;
    }
}

template class InputTransform<4, 4, __fp16, __fp16, WinogradRoots::Integers>;

}  // namespace
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
