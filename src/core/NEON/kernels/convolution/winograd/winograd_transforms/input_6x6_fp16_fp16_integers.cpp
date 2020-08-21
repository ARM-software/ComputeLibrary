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
#include "arm.hpp"
#include "input.hpp"

namespace winograd
{
template <>
void InputTransform<6, 6, __fp16, __fp16, WinogradRoots::Integers>::transform_tile(
    const int n_channels,
    const __fp16* const input_base,
    const int input_row_stride,
    const int input_col_stride,
    __fp16* outptr,
    const int matrix_stride
)
{
    constexpr int inner_tile_rows = 6;
    constexpr int inner_tile_cols = 6;

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
    for (; channels_remaining >= 8; channels_remaining -= 8)
    {
        // Matrices used/computed in this kernel
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

        // Read a 6x6 tile in the Winograd domain
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
            // XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
            XTx[0][j] = vsubq_f16(vaddq_f16(x[4][j], vmulq_f16(x[0][j], vdupq_n_f16(4.0f))), vmulq_f16(x[2][j], vdupq_n_f16(5.0f)));

            // XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
            XTx[1][j] = vsubq_f16(vaddq_f16(x[3][j], x[4][j]), vmulq_f16(vaddq_f16(x[1][j], x[2][j]),  vdupq_n_f16(4.0f)));

            // XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
            XTx[2][j] = vaddq_f16(vsubq_f16(x[4][j], x[3][j]), vmulq_f16(vsubq_f16(x[1][j], x[2][j]), vdupq_n_f16(4.0f)));

            // XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
            XTx[3][j] = vaddq_f16(vsubq_f16(x[4][j], x[2][j]), vmulq_f16(vsubq_f16(x[3][j], x[1][j]), vdupq_n_f16(2.0f)));

            // XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
            XTx[4][j] = vaddq_f16(vsubq_f16(x[4][j], x[2][j]), vmulq_f16(vsubq_f16(x[1][j], x[3][j]), vdupq_n_f16(2.0f)));

            // XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
            XTx[5][j] = vsubq_f16(vaddq_f16(x[5][j], vmulq_f16(x[1][j], vdupq_n_f16(4.0f))), vmulq_f16(x[3][j], vdupq_n_f16(5.0f)));
        }

        // Compute U = XT . x . X
        for (int i = 0; i < inner_tile_rows; i++)
        {
            // U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
            U[i][0] = vsubq_f16(vaddq_f16(XTx[i][4], vmulq_f16(XTx[i][0], vdupq_n_f16(4.0f))), vmulq_f16(XTx[i][2], vdupq_n_f16(5.0f)));

            // U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
            U[i][1] = vsubq_f16(vaddq_f16(XTx[i][3], XTx[i][4]), vmulq_f16(vaddq_f16(XTx[i][1], XTx[i][2]), vdupq_n_f16(4.0f)));

            // U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
            U[i][2] = vaddq_f16(vsubq_f16(XTx[i][4], XTx[i][3]), vmulq_f16(vsubq_f16(XTx[i][1], XTx[i][2]), vdupq_n_f16(4.0f)));

            // U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
            U[i][3] = vaddq_f16(vsubq_f16(XTx[i][4], XTx[i][2]), vmulq_f16(vsubq_f16(XTx[i][3], XTx[i][1]), vdupq_n_f16(2.0f)));

            // U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
            U[i][4] = vaddq_f16(vsubq_f16(XTx[i][4], XTx[i][2]), vmulq_f16(vsubq_f16(XTx[i][1], XTx[i][3]), vdupq_n_f16(2.0f)));

            // U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
            U[i][5] = vsubq_f16(vaddq_f16(XTx[i][5], vmulq_f16(XTx[i][1], vdupq_n_f16(4.0f))), vmulq_f16(XTx[i][3], vdupq_n_f16(5.0f)));
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
    for (; channels_remaining >= 4; channels_remaining -= 4)
    {
        // Matrices used/computed in this kernel
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

        // Read a 6x6 tile in the Winograd domain
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
            // XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
            XTx[0][j] = vsub_f16(vadd_f16(x[4][j], vmul_f16(x[0][j], vdup_n_f16(4.0f))), vmul_f16(x[2][j], vdup_n_f16(5.0f)));

            // XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
            XTx[1][j] = vsub_f16(vadd_f16(x[3][j], x[4][j]), vmul_f16(vadd_f16(x[1][j], x[2][j]),  vdup_n_f16(4.0f)));

            // XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
            XTx[2][j] = vadd_f16(vsub_f16(x[4][j], x[3][j]), vmul_f16(vsub_f16(x[1][j], x[2][j]), vdup_n_f16(4.0f)));

            // XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
            XTx[3][j] = vadd_f16(vsub_f16(x[4][j], x[2][j]), vmul_f16(vsub_f16(x[3][j], x[1][j]), vdup_n_f16(2.0f)));

            // XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
            XTx[4][j] = vadd_f16(vsub_f16(x[4][j], x[2][j]), vmul_f16(vsub_f16(x[1][j], x[3][j]), vdup_n_f16(2.0f)));

            // XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
            XTx[5][j] = vsub_f16(vadd_f16(x[5][j], vmul_f16(x[1][j], vdup_n_f16(4.0f))), vmul_f16(x[3][j], vdup_n_f16(5.0f)));
        }

        // Compute U = XT . x . X
        for (int i = 0; i < inner_tile_rows; i++)
        {
            // U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
            U[i][0] = vsub_f16(vadd_f16(XTx[i][4], vmul_f16(XTx[i][0], vdup_n_f16(4.0f))), vmul_f16(XTx[i][2], vdup_n_f16(5.0f)));

            // U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
            U[i][1] = vsub_f16(vadd_f16(XTx[i][3], XTx[i][4]), vmul_f16(vadd_f16(XTx[i][1], XTx[i][2]), vdup_n_f16(4.0f)));

            // U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
            U[i][2] = vadd_f16(vsub_f16(XTx[i][4], XTx[i][3]), vmul_f16(vsub_f16(XTx[i][1], XTx[i][2]), vdup_n_f16(4.0f)));

            // U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
            U[i][3] = vadd_f16(vsub_f16(XTx[i][4], XTx[i][2]), vmul_f16(vsub_f16(XTx[i][3], XTx[i][1]), vdup_n_f16(2.0f)));

            // U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
            U[i][4] = vadd_f16(vsub_f16(XTx[i][4], XTx[i][2]), vmul_f16(vsub_f16(XTx[i][1], XTx[i][3]), vdup_n_f16(2.0f)));

            // U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
            U[i][5] = vsub_f16(vadd_f16(XTx[i][5], vmul_f16(XTx[i][1], vdup_n_f16(4.0f))), vmul_f16(XTx[i][3], vdup_n_f16(5.0f)));
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
            XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
            XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
            XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
            XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
            XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
            XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
        }

        // Compute U = XT . x . X
        for (int i = 0; i < inner_tile_rows; i++)
        {
            U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
            U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
            U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
            U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
            U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
            U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
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

template class InputTransform<6, 6, __fp16, __fp16, WinogradRoots::Integers>;

}  // namespace winograd
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC