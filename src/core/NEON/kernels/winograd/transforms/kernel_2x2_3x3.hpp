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
#pragma once

namespace winograd {
  /* Transform a kernel into the Winograd domain.
   *
   * NOTE: It is assumed that the kernel is in the form [height x width x
   * input_channels x output_channel].
   */
  template <typename T>
  struct winograd2x2_3x3_gemm_kernel_transform_impl{
    static void execute(
      const KernelShape &shape,
      const T* const kernel,
      T* const matrix_base,
      const int matrix_stride,
      const int matrix_row_stride
    );

    protected:
    template <const int output_channel_tail>
    static void transform_kernel(
      const T* const kernel,
      const int n_input_channels,
      const int n_output_channels,
      T* const matrix_base,
      const int matrix_stride,
      const int matrix_row_stride
    );
  };
}

/*****************************************************************************/
/* Transform a fp32 kernel into the Winograd domain.
 */
#include "kernel_2x2_3x3/a64_float.hpp"  // AArch64 specialisations

namespace winograd
{
template <>
inline void winograd2x2_3x3_gemm_kernel_transform_impl<float>::execute(
  const KernelShape &shape,
  const float* const kernel,
  float* const matrix_base,
  const int matrix_stride,
  const int matrix_row_stride
) {
  // Delegate based on tail size
  const int n_input_channels = shape.n_input_channels;
  const int n_output_channels = shape.n_output_channels;

  switch (n_output_channels % 4) {
    case 0:
      transform_kernel<0>(
        kernel, n_input_channels, n_output_channels,
        matrix_base, matrix_stride, matrix_row_stride
      );
      break;
    case 1:
      transform_kernel<1>(
        kernel, n_input_channels, n_output_channels,
        matrix_base, matrix_stride, matrix_row_stride
      );
      break;
    case 2:
      transform_kernel<2>(
        kernel, n_input_channels, n_output_channels,
        matrix_base, matrix_stride, matrix_row_stride
      );
      break;
    case 3:
      transform_kernel<3>(
        kernel, n_input_channels, n_output_channels,
        matrix_base, matrix_stride, matrix_row_stride
      );
      break;
    default:
        ARM_COMPUTE_ERROR("Cannot happen");
        break;
  }
}

template <>
template<const int output_channel_tail>
inline void winograd2x2_3x3_gemm_kernel_transform_impl<float>::transform_kernel(
    const float* const kernel,
    const int n_input_channels,
    const int n_output_channels,
    float* const matrix_base,
    const int mstride,
    const int matrix_row_stride
) {
  // Use one input pointer for each row of the kernel, use two additional
  // offsets to extract columns.
  const int kernel_col_stride = n_input_channels * n_output_channels;
  const int kernel_row_stride = 3 * kernel_col_stride;
  const float *inptr0 = kernel;
  const float *inptr1 = kernel + kernel_row_stride;
  const float *inptr2 = kernel + kernel_row_stride*2;

  // Use four output pointers, for output matrices 0, 4, 8 and 12. Use three
  // offsets to extract further matrices.
  float  *outptr0 = matrix_base;
  float  *outptr4 = matrix_base + mstride * 4;
  float  *outptr8 = matrix_base + mstride * 8;
  float *outptr12 = matrix_base + mstride * 12;

  // For every input channel
  for (int in_c = 0; in_c < n_input_channels; in_c++) {
    // For every output channel
    for (int c = 0; c < n_output_channels; c++) {
      // Read in the kernel
      float w11 = inptr0[0], w12 = inptr0[kernel_col_stride], w13 = inptr0[kernel_col_stride*2];
      float w21 = inptr1[0], w22 = inptr1[kernel_col_stride], w23 = inptr1[kernel_col_stride*2];
      float w31 = inptr2[0], w32 = inptr2[kernel_col_stride], w33 = inptr2[kernel_col_stride*2];

      // Progress input pointers
      inptr0++;
      inptr1++;
      inptr2++;

      // Compute the kernel W w, note we need only compute the middle two rows
      // (2 and 3) because the first and last rows are merely copies of values
      // from the matrix w.
      float Ww11 = w11, Ww12 = w12, Ww13 = w13;
      float Ww21 = 0.5*(w11 + w21 + w31), Ww22 = 0.5*(w12 + w22 + w32), Ww23 = 0.5*(w13 + w23 + w33);
      float Ww31 = 0.5*(w11 - w21 + w31), Ww32 = 0.5*(w12 - w22 + w32), Ww33 = 0.5*(w13 - w23 + w33);
      float Ww41 = w31, Ww42 = w32, Ww43 = w33;

      // Hence compute W w W.T; again note we need compute only the middle two
      // columns since the first and last columns are copies of the first and
      // last columns of the previous matrix.
      float WwWT11 = Ww11, WwWT12 = 0.5*(Ww11 + Ww12 + Ww13), WwWT13 = 0.5*(Ww11 - Ww12 + Ww13), WwWT14 = Ww13;
      float WwWT21 = Ww21, WwWT22 = 0.5*(Ww21 + Ww22 + Ww23), WwWT23 = 0.5*(Ww21 - Ww22 + Ww23), WwWT24 = Ww23;
      float WwWT31 = Ww31, WwWT32 = 0.5*(Ww31 + Ww32 + Ww33), WwWT33 = 0.5*(Ww31 - Ww32 + Ww33), WwWT34 = Ww33;
      float WwWT41 = Ww41, WwWT42 = 0.5*(Ww41 + Ww42 + Ww43), WwWT43 = 0.5*(Ww41 - Ww42 + Ww43), WwWT44 = Ww43;

      // Store the computed weights
      outptr0[0 * mstride] = WwWT11;
      outptr0[1 * mstride] = WwWT12;
      outptr0[2 * mstride] = WwWT13;
      outptr0[3 * mstride] = WwWT14;

      outptr4[0 * mstride] = WwWT21;
      outptr4[1 * mstride] = WwWT22;
      outptr4[2 * mstride] = WwWT23;
      outptr4[3 * mstride] = WwWT24;

      outptr8[0 * mstride] = WwWT31;
      outptr8[1 * mstride] = WwWT32;
      outptr8[2 * mstride] = WwWT33;
      outptr8[3 * mstride] = WwWT34;

      outptr12[0 * mstride] = WwWT41;
      outptr12[1 * mstride] = WwWT42;
      outptr12[2 * mstride] = WwWT43;
      outptr12[3 * mstride] = WwWT44;

      // Progress output pointers
      outptr0++;
      outptr4++;
      outptr8++;
      outptr12++;
    }

    // Progression to complete stride
    outptr0 += matrix_row_stride - n_output_channels;
    outptr4 += matrix_row_stride - n_output_channels;
    outptr8 += matrix_row_stride - n_output_channels;
    outptr12 += matrix_row_stride - n_output_channels;
  }
}
}
