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

#ifdef __aarch64__
namespace winograd {
template <>
template <>
inline void winograd2x2_3x3_gemm_kernel_transform_impl<float>::transform_kernel<0>(
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
    int n_remaining_channels = n_output_channels;

    asm volatile (
        // Registers into which to read the kernel
        "w_11 .req v0\n"  "qw_11 .req q0\n"
        "w_12 .req v1\n"  "qw_12 .req q1\n"
        "w_13 .req v2\n"  "qw_13 .req q2\n"
        "w_21 .req v3\n"  "qw_21 .req q3\n"
        "w_22 .req v4\n"  "qw_22 .req q4\n"
        "w_23 .req v5\n"  "qw_23 .req q5\n"
        "w_31 .req v6\n"  "qw_31 .req q6\n"
        "w_32 .req v7\n"  "qw_32 .req q7\n"
        "w_33 .req v8\n"  "qw_33 .req q8\n"

        // Transformed matrix Ww
        "Ww11 .req w_11\n"  "Ww12 .req w_12\n"  "Ww13 .req w_13\n"
        "Ww21 .req  v9\n"   "Ww22 .req v10\n"   "Ww23 .req v11\n"
        "Ww31 .req v12\n"   "Ww32 .req v13\n"   "Ww33 .req v14\n"
        "Ww41 .req w_31\n"  "Ww42 .req w_32\n"  "Ww43 .req w_33\n"

        // Output matrix U = WwWT
        "U11 .req Ww11\n"   "U12 .req v15\n"  "U13 .req v16\n"  "U14 .req Ww13\n"
        "U21 .req Ww21\n"   "U22 .req v17\n"  "U23 .req v18\n"  "U24 .req Ww23\n"
        "U31 .req Ww31\n"   "U32 .req v19\n"  "U33 .req v20\n"  "U34 .req Ww33\n"
        "U41 .req Ww41\n"   "U42 .req v21\n"  "U43 .req v22\n"  "U44 .req Ww43\n"

        // Storage view of output matrices
        "qU11 .req   q0\n"   "qU12 .req q15\n"  "qU13 .req q16\n"  "qU14 .req   q2\n"
        "qU21 .req   q9\n"   "qU22 .req q17\n"  "qU23 .req q18\n"  "qU24 .req  q11\n"
        "qU31 .req  q12\n"   "qU32 .req q19\n"  "qU33 .req q20\n"  "qU34 .req  q14\n"
        "qU41 .req   q6\n"   "qU42 .req q21\n"  "qU43 .req q22\n"  "qU44 .req   q8\n"

        "half .req v23\n"  // {0.5, ..., 0.5}
        "dup half.4s, %w[one_half]\n"
        "scratch .req v24\n"

        "1:"
          // Load tile of the kernel
          "ldr qw_11, [%x[inptr0]]\n"
          "str qU11, [%x[outptr0]]\n"
          "ldr qw_12, [%x[inptr0], %x[colstride1]]\n"
          "ldr qw_13, [%x[inptr0], %x[colstride2]]\n"
          "str qU14, [%x[outptr0], %x[mstride3]]\n"
          "add %x[inptr0], %x[inptr0], #0x10\n"

          "ldr qw_21, [%x[inptr1]]\n"
          "ldr qw_22, [%x[inptr1], %x[colstride1]]\n"
          "ldr qw_23, [%x[inptr1], %x[colstride2]]\n"
          "add %x[inptr1], %x[inptr1], #0x10\n"

          "ldr qw_31, [%x[inptr2]]\n"
          "str qU41, [%x[outptr12]]\n"
          "ldr qw_32, [%x[inptr2], %x[colstride1]]\n"
          "ldr qw_33, [%x[inptr2], %x[colstride2]]\n"
          "str qU44, [%x[outptr12], %x[mstride3]]\n"
          "add %x[inptr2], %x[inptr2], #0x10\n"

          // Compute 2nd and 3rd rows of Ww
          "fadd scratch.4s, w_11.4s, w_31.4s\n"
          "fmul Ww21.4s, scratch.4s, half.4s\n"
          "fmla Ww21.4s, w_21.4s, half.4s\n"
          "str qU21, [%x[outptr4]]\n"
          "fmul Ww31.4s, scratch.4s, half.4s\n"
          "fmls Ww31.4s, w_21.4s, half.4s\n"
          "str qU31, [%x[outptr8]]\n"

          "fadd scratch.4s, w_12.4s, w_32.4s\n"
          "fmul Ww22.4s, scratch.4s, half.4s\n"
          "fmla Ww22.4s, w_22.4s, half.4s\n"
          "fmul Ww32.4s, scratch.4s, half.4s\n"
          "fmls Ww32.4s, w_22.4s, half.4s\n"

          "fadd scratch.4s, w_13.4s, w_33.4s\n"
          "fmul Ww23.4s, scratch.4s, half.4s\n"
          "fmla Ww23.4s, w_23.4s, half.4s\n"
          "str qU24, [%x[outptr4], %x[mstride3]]\n"
          "fmul Ww33.4s, scratch.4s, half.4s\n"
          "fmls Ww33.4s, w_23.4s, half.4s\n"
          "str qU34, [%x[outptr8], %x[mstride3]]\n"

          // Compute and store U, only need to compute the 2nd and 3rd columns
          // of U and update output pointers
          "fadd scratch.4s, Ww11.4s, Ww13.4s\n"
          "fmul U12.4s, scratch.4s, half.4s\n"
          "fmla U12.4s, Ww12.4s, half.4s\n"
          "str qU12, [%x[outptr0], %x[mstride1]]\n"
          "fmul U13.4s, scratch.4s, half.4s\n"
          "fmls U13.4s, Ww12.4s, half.4s\n"
          "str qU13, [%x[outptr0], %x[mstride2]]\n"
          "add  %x[outptr0],  %x[outptr0], #0x10\n"

          "fadd scratch.4s, Ww21.4s, Ww23.4s\n"
          "fmul U22.4s, scratch.4s, half.4s\n"
          "fmla U22.4s, Ww22.4s, half.4s\n"
          "str qU22, [%x[outptr4], %x[mstride1]]\n"
          "fmul U23.4s, scratch.4s, half.4s\n"
          "fmls U23.4s, Ww22.4s, half.4s\n"
          "str qU23, [%x[outptr4], %x[mstride2]]\n"
          "add  %x[outptr4],  %x[outptr4], #0x10\n"

          "fadd scratch.4s, Ww31.4s, Ww33.4s\n"
          "fmul U32.4s, scratch.4s, half.4s\n"
          "fmla U32.4s, Ww32.4s, half.4s\n"
          "str qU32, [%x[outptr8], %x[mstride1]]\n"
          "fmul U33.4s, scratch.4s, half.4s\n"
          "fmls U33.4s, Ww32.4s, half.4s\n"
          "str qU33, [%x[outptr8], %x[mstride2]]\n"
          "add  %x[outptr8],  %x[outptr8], #0x10\n"

          "fadd scratch.4s, Ww41.4s, Ww43.4s\n"
          "fmul U42.4s, scratch.4s, half.4s\n"
          "fmla U42.4s, Ww42.4s, half.4s\n"
          "str qU42, [%x[outptr12], %x[mstride1]]\n"
          "fmul U43.4s, scratch.4s, half.4s\n"
          "fmls U43.4s, Ww42.4s, half.4s\n"
          "str qU43, [%x[outptr12], %x[mstride2]]\n"
          "add %x[outptr12], %x[outptr12], #0x10\n"

          "subs %x[n_remaining_channels], %x[n_remaining_channels], #4\n"
          "bne 1b\n"

        // Clear aliases
        ".unreq half\n"
        ".unreq scratch\n"
        ".unreq w_11\n"  ".unreq qw_11\n"
        ".unreq w_12\n"  ".unreq qw_12\n"
        ".unreq w_13\n"  ".unreq qw_13\n"
        ".unreq w_21\n"  ".unreq qw_21\n"
        ".unreq w_22\n"  ".unreq qw_22\n"
        ".unreq w_23\n"  ".unreq qw_23\n"
        ".unreq w_31\n"  ".unreq qw_31\n"
        ".unreq w_32\n"  ".unreq qw_32\n"
        ".unreq w_33\n"  ".unreq qw_33\n"
        ".unreq Ww11\n"  ".unreq Ww12\n"  ".unreq Ww13\n"
        ".unreq Ww21\n"  ".unreq Ww22\n"  ".unreq Ww23\n"
        ".unreq Ww31\n"  ".unreq Ww32\n"  ".unreq Ww33\n"
        ".unreq Ww41\n"  ".unreq Ww42\n"  ".unreq Ww43\n"
        ".unreq U11\n"   ".unreq U12\n"   ".unreq U13\n"   ".unreq U14\n"
        ".unreq U21\n"   ".unreq U22\n"   ".unreq U23\n"   ".unreq U24\n"
        ".unreq U31\n"   ".unreq U32\n"   ".unreq U33\n"   ".unreq U34\n"
        ".unreq U41\n"   ".unreq U42\n"   ".unreq U43\n"   ".unreq U44\n"
        ".unreq qU11\n"  ".unreq qU12\n"  ".unreq qU13\n"  ".unreq qU14\n"
        ".unreq qU21\n"  ".unreq qU22\n"  ".unreq qU23\n"  ".unreq qU24\n"
        ".unreq qU31\n"  ".unreq qU32\n"  ".unreq qU33\n"  ".unreq qU34\n"
        ".unreq qU41\n"  ".unreq qU42\n"  ".unreq qU43\n"  ".unreq qU44\n"

      : [inptr0] "+r" (inptr0),
        [inptr1] "+r" (inptr1),
        [inptr2] "+r" (inptr2),
        [outptr0] "+r" (outptr0),
        [outptr4] "+r" (outptr4),
        [outptr8] "+r" (outptr8),
        [outptr12] "+r" (outptr12),
        [n_remaining_channels] "+r" (n_remaining_channels)
      : [mstride1] "r" (sizeof(float) * mstride),
        [mstride2] "r" (sizeof(float) * mstride * 2),
        [mstride3] "r" (sizeof(float) * mstride * 3),
        [colstride1] "r" (sizeof(float) * kernel_col_stride),
        [colstride2] "r" (sizeof(float) * kernel_col_stride * 2),
        [one_half] "r" (0.5f)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24"
    );

    // Progression to complete stride
    outptr0 += matrix_row_stride - n_output_channels;
    outptr4 += matrix_row_stride - n_output_channels;
    outptr8 += matrix_row_stride - n_output_channels;
    outptr12 += matrix_row_stride - n_output_channels;
  }
}

template <>
template <>
inline void winograd2x2_3x3_gemm_kernel_transform_impl<float>::transform_kernel<2>(
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
    int n_remaining_channels = n_output_channels;

    asm volatile (
        // Registers into which to read the kernel
        "w_11 .req v0\n"  "qw_11 .req q0\n"  "dw_11 .req d0\n"
        "w_12 .req v1\n"  "qw_12 .req q1\n"  "dw_12 .req d1\n"
        "w_13 .req v2\n"  "qw_13 .req q2\n"  "dw_13 .req d2\n"
        "w_21 .req v3\n"  "qw_21 .req q3\n"  "dw_21 .req d3\n"
        "w_22 .req v4\n"  "qw_22 .req q4\n"  "dw_22 .req d4\n"
        "w_23 .req v5\n"  "qw_23 .req q5\n"  "dw_23 .req d5\n"
        "w_31 .req v6\n"  "qw_31 .req q6\n"  "dw_31 .req d6\n"
        "w_32 .req v7\n"  "qw_32 .req q7\n"  "dw_32 .req d7\n"
        "w_33 .req v8\n"  "qw_33 .req q8\n"  "dw_33 .req d8\n"

        // Transformed matrix Ww
        "Ww11 .req w_11\n"  "Ww12 .req w_12\n"  "Ww13 .req w_13\n"
        "Ww21 .req  v9\n"   "Ww22 .req v10\n"   "Ww23 .req v11\n"
        "Ww31 .req v12\n"   "Ww32 .req v13\n"   "Ww33 .req v14\n"
        "Ww41 .req w_31\n"  "Ww42 .req w_32\n"  "Ww43 .req w_33\n"

        // Output matrix U = WwWT
        "U11 .req Ww11\n"   "U12 .req v15\n"  "U13 .req v16\n"  "U14 .req Ww13\n"
        "U21 .req Ww21\n"   "U22 .req v17\n"  "U23 .req v18\n"  "U24 .req Ww23\n"
        "U31 .req Ww31\n"   "U32 .req v19\n"  "U33 .req v20\n"  "U34 .req Ww33\n"
        "U41 .req Ww41\n"   "U42 .req v21\n"  "U43 .req v22\n"  "U44 .req Ww43\n"

        // Storage view of output matrices
        "qU11 .req   q0\n"   "qU12 .req q15\n"  "qU13 .req q16\n"  "qU14 .req   q2\n"
        "qU21 .req   q9\n"   "qU22 .req q17\n"  "qU23 .req q18\n"  "qU24 .req  q11\n"
        "qU31 .req  q12\n"   "qU32 .req q19\n"  "qU33 .req q20\n"  "qU34 .req  q14\n"
        "qU41 .req   q6\n"   "qU42 .req q21\n"  "qU43 .req q22\n"  "qU44 .req   q8\n"

        "dU11 .req   d0\n"   "dU12 .req d15\n"  "dU13 .req d16\n"  "dU14 .req   d2\n"
        "dU21 .req   d9\n"   "dU22 .req d17\n"  "dU23 .req d18\n"  "dU24 .req  d11\n"
        "dU31 .req  d12\n"   "dU32 .req d19\n"  "dU33 .req d20\n"  "dU34 .req  d14\n"
        "dU41 .req   d6\n"   "dU42 .req d21\n"  "dU43 .req d22\n"  "dU44 .req   d8\n"

        "half .req v23\n"  // {0.5, ..., 0.5}
        "dup half.4s, %w[one_half]\n"
        "scratch .req v24\n"
        
        // Subtract the tail from the number of remaining channels and jump to
        // the tail if necessary.
        "subs %x[n_remaining_channels], %x[n_remaining_channels], #2\n"
        "beq 2f\n"

        "1:"
          // Load tile of the kernel
          "ldr qw_11, [%x[inptr0]]\n"
          "str qU11, [%x[outptr0]]\n"
          "ldr qw_12, [%x[inptr0], %x[colstride1]]\n"
          "ldr qw_13, [%x[inptr0], %x[colstride2]]\n"
          "str qU14, [%x[outptr0], %x[mstride3]]\n"
          "add %x[inptr0], %x[inptr0], #0x10\n"

          "ldr qw_21, [%x[inptr1]]\n"
          "ldr qw_22, [%x[inptr1], %x[colstride1]]\n"
          "ldr qw_23, [%x[inptr1], %x[colstride2]]\n"
          "add %x[inptr1], %x[inptr1], #0x10\n"

          "ldr qw_31, [%x[inptr2]]\n"
          "str qU41, [%x[outptr12]]\n"
          "ldr qw_32, [%x[inptr2], %x[colstride1]]\n"
          "ldr qw_33, [%x[inptr2], %x[colstride2]]\n"
          "str qU44, [%x[outptr12], %x[mstride3]]\n"
          "add %x[inptr2], %x[inptr2], #0x10\n"

          // Compute 2nd and 3rd rows of Ww
          "fadd scratch.4s, w_11.4s, w_31.4s\n"
          "fmul Ww21.4s, scratch.4s, half.4s\n"
          "fmla Ww21.4s, w_21.4s, half.4s\n"
          "str qU21, [%x[outptr4]]\n"
          "fmul Ww31.4s, scratch.4s, half.4s\n"
          "fmls Ww31.4s, w_21.4s, half.4s\n"
          "str qU31, [%x[outptr8]]\n"

          "fadd scratch.4s, w_12.4s, w_32.4s\n"
          "fmul Ww22.4s, scratch.4s, half.4s\n"
          "fmla Ww22.4s, w_22.4s, half.4s\n"
          "fmul Ww32.4s, scratch.4s, half.4s\n"
          "fmls Ww32.4s, w_22.4s, half.4s\n"

          "fadd scratch.4s, w_13.4s, w_33.4s\n"
          "fmul Ww23.4s, scratch.4s, half.4s\n"
          "fmla Ww23.4s, w_23.4s, half.4s\n"
          "str qU24, [%x[outptr4], %x[mstride3]]\n"
          "fmul Ww33.4s, scratch.4s, half.4s\n"
          "fmls Ww33.4s, w_23.4s, half.4s\n"
          "str qU34, [%x[outptr8], %x[mstride3]]\n"

          // Compute and store U, only need to compute the 2nd and 3rd columns
          // of U and update output pointers
          "fadd scratch.4s, Ww11.4s, Ww13.4s\n"
          "fmul U12.4s, scratch.4s, half.4s\n"
          "fmla U12.4s, Ww12.4s, half.4s\n"
          "str qU12, [%x[outptr0], %x[mstride1]]\n"
          "fmul U13.4s, scratch.4s, half.4s\n"
          "fmls U13.4s, Ww12.4s, half.4s\n"
          "str qU13, [%x[outptr0], %x[mstride2]]\n"
          "add  %x[outptr0],  %x[outptr0], #0x10\n"

          "fadd scratch.4s, Ww21.4s, Ww23.4s\n"
          "fmul U22.4s, scratch.4s, half.4s\n"
          "fmla U22.4s, Ww22.4s, half.4s\n"
          "str qU22, [%x[outptr4], %x[mstride1]]\n"
          "fmul U23.4s, scratch.4s, half.4s\n"
          "fmls U23.4s, Ww22.4s, half.4s\n"
          "str qU23, [%x[outptr4], %x[mstride2]]\n"
          "add  %x[outptr4],  %x[outptr4], #0x10\n"

          "fadd scratch.4s, Ww31.4s, Ww33.4s\n"
          "fmul U32.4s, scratch.4s, half.4s\n"
          "fmla U32.4s, Ww32.4s, half.4s\n"
          "str qU32, [%x[outptr8], %x[mstride1]]\n"
          "fmul U33.4s, scratch.4s, half.4s\n"
          "fmls U33.4s, Ww32.4s, half.4s\n"
          "str qU33, [%x[outptr8], %x[mstride2]]\n"
          "add  %x[outptr8],  %x[outptr8], #0x10\n"

          "fadd scratch.4s, Ww41.4s, Ww43.4s\n"
          "fmul U42.4s, scratch.4s, half.4s\n"
          "fmla U42.4s, Ww42.4s, half.4s\n"
          "str qU42, [%x[outptr12], %x[mstride1]]\n"
          "fmul U43.4s, scratch.4s, half.4s\n"
          "fmls U43.4s, Ww42.4s, half.4s\n"
          "str qU43, [%x[outptr12], %x[mstride2]]\n"
          "add %x[outptr12], %x[outptr12], #0x10\n"

          "subs %x[n_remaining_channels], %x[n_remaining_channels], #4\n"
          "bne 1b\n"

        // Tail size 2
        "2:"
          // Load tile of the kernel
          "ldr dw_11, [%x[inptr0]]\n"
          "str dU11, [%x[outptr0]]\n"
          "ldr dw_12, [%x[inptr0], %x[colstride1]]\n"
          "ldr dw_13, [%x[inptr0], %x[colstride2]]\n"
          "str dU14, [%x[outptr0], %x[mstride3]]\n"
          "add %x[inptr0], %x[inptr0], #0x08\n"

          "ldr dw_21, [%x[inptr1]]\n"
          "ldr dw_22, [%x[inptr1], %x[colstride1]]\n"
          "ldr dw_23, [%x[inptr1], %x[colstride2]]\n"
          "add %x[inptr1], %x[inptr1], #0x08\n"

          "ldr dw_31, [%x[inptr2]]\n"
          "str dU41, [%x[outptr12]]\n"
          "ldr dw_32, [%x[inptr2], %x[colstride1]]\n"
          "ldr dw_33, [%x[inptr2], %x[colstride2]]\n"
          "str dU44, [%x[outptr12], %x[mstride3]]\n"
          "add %x[inptr2], %x[inptr2], #0x08\n"

          // Compute 2nd and 3rd rows of Ww
          "fadd scratch.2s, w_11.2s, w_31.2s\n"
          "fmul Ww21.2s, scratch.2s, half.2s\n"
          "fmla Ww21.2s, w_21.2s, half.2s\n"
          "str dU21, [%x[outptr4]]\n"
          "fmul Ww31.2s, scratch.2s, half.2s\n"
          "fmls Ww31.2s, w_21.2s, half.2s\n"
          "str dU31, [%x[outptr8]]\n"

          "fadd scratch.2s, w_12.2s, w_32.2s\n"
          "fmul Ww22.2s, scratch.2s, half.2s\n"
          "fmla Ww22.2s, w_22.2s, half.2s\n"
          "fmul Ww32.2s, scratch.2s, half.2s\n"
          "fmls Ww32.2s, w_22.2s, half.2s\n"

          "fadd scratch.2s, w_13.2s, w_33.2s\n"
          "fmul Ww23.2s, scratch.2s, half.2s\n"
          "fmla Ww23.2s, w_23.2s, half.2s\n"
          "str dU24, [%x[outptr4], %x[mstride3]]\n"
          "fmul Ww33.2s, scratch.2s, half.2s\n"
          "fmls Ww33.2s, w_23.2s, half.2s\n"
          "str dU34, [%x[outptr8], %x[mstride3]]\n"

          // Compute and store U, only need to compute the 2nd and 3rd columns of
          // U and update output pointers
          "fadd scratch.2s, Ww11.2s, Ww13.2s\n"
          "fmul U12.2s, scratch.2s, half.2s\n"
          "fmla U12.2s, Ww12.2s, half.2s\n"
          "str dU12, [%x[outptr0], %x[mstride1]]\n"
          "fmul U13.2s, scratch.2s, half.2s\n"
          "fmls U13.2s, Ww12.2s, half.2s\n"
          "str dU13, [%x[outptr0], %x[mstride2]]\n"
          "add  %x[outptr0],  %x[outptr0], #0x08\n"

          "fadd scratch.2s, Ww21.2s, Ww23.2s\n"
          "fmul U22.2s, scratch.2s, half.2s\n"
          "fmla U22.2s, Ww22.2s, half.2s\n"
          "str dU22, [%x[outptr4], %x[mstride1]]\n"
          "fmul U23.2s, scratch.2s, half.2s\n"
          "fmls U23.2s, Ww22.2s, half.2s\n"
          "str dU23, [%x[outptr4], %x[mstride2]]\n"
          "add  %x[outptr4],  %x[outptr4], #0x08\n"

          "fadd scratch.2s, Ww31.2s, Ww33.2s\n"
          "fmul U32.2s, scratch.2s, half.2s\n"
          "fmla U32.2s, Ww32.2s, half.2s\n"
          "str dU32, [%x[outptr8], %x[mstride1]]\n"
          "fmul U33.2s, scratch.2s, half.2s\n"
          "fmls U33.2s, Ww32.2s, half.2s\n"
          "str dU33, [%x[outptr8], %x[mstride2]]\n"
          "add  %x[outptr8],  %x[outptr8], #0x08\n"

          "fadd scratch.2s, Ww41.2s, Ww43.2s\n"
          "fmul U42.2s, scratch.2s, half.2s\n"
          "fmla U42.2s, Ww42.2s, half.2s\n"
          "str dU42, [%x[outptr12], %x[mstride1]]\n"
          "fmul U43.2s, scratch.2s, half.2s\n"
          "fmls U43.2s, Ww42.2s, half.2s\n"
          "str dU43, [%x[outptr12], %x[mstride2]]\n"
          "add %x[outptr12], %x[outptr12], #0x08\n"

        // Clear aliases
        ".unreq half\n"
        ".unreq scratch\n"
        ".unreq w_11\n"  ".unreq qw_11\n" ".unreq dw_11\n"
        ".unreq w_12\n"  ".unreq qw_12\n" ".unreq dw_12\n"
        ".unreq w_13\n"  ".unreq qw_13\n" ".unreq dw_13\n"
        ".unreq w_21\n"  ".unreq qw_21\n" ".unreq dw_21\n"
        ".unreq w_22\n"  ".unreq qw_22\n" ".unreq dw_22\n"
        ".unreq w_23\n"  ".unreq qw_23\n" ".unreq dw_23\n"
        ".unreq w_31\n"  ".unreq qw_31\n" ".unreq dw_31\n"
        ".unreq w_32\n"  ".unreq qw_32\n" ".unreq dw_32\n"
        ".unreq w_33\n"  ".unreq qw_33\n" ".unreq dw_33\n"
        ".unreq Ww11\n"  ".unreq Ww12\n"  ".unreq Ww13\n"
        ".unreq Ww21\n"  ".unreq Ww22\n"  ".unreq Ww23\n"
        ".unreq Ww31\n"  ".unreq Ww32\n"  ".unreq Ww33\n"
        ".unreq Ww41\n"  ".unreq Ww42\n"  ".unreq Ww43\n"
        ".unreq U11\n"   ".unreq U12\n"   ".unreq U13\n"   ".unreq U14\n"
        ".unreq U21\n"   ".unreq U22\n"   ".unreq U23\n"   ".unreq U24\n"
        ".unreq U31\n"   ".unreq U32\n"   ".unreq U33\n"   ".unreq U34\n"
        ".unreq U41\n"   ".unreq U42\n"   ".unreq U43\n"   ".unreq U44\n"
        ".unreq qU11\n"  ".unreq qU12\n"  ".unreq qU13\n"  ".unreq qU14\n"
        ".unreq qU21\n"  ".unreq qU22\n"  ".unreq qU23\n"  ".unreq qU24\n"
        ".unreq qU31\n"  ".unreq qU32\n"  ".unreq qU33\n"  ".unreq qU34\n"
        ".unreq qU41\n"  ".unreq qU42\n"  ".unreq qU43\n"  ".unreq qU44\n"
        ".unreq dU11\n"  ".unreq dU12\n"  ".unreq dU13\n"  ".unreq dU14\n"
        ".unreq dU21\n"  ".unreq dU22\n"  ".unreq dU23\n"  ".unreq dU24\n"
        ".unreq dU31\n"  ".unreq dU32\n"  ".unreq dU33\n"  ".unreq dU34\n"
        ".unreq dU41\n"  ".unreq dU42\n"  ".unreq dU43\n"  ".unreq dU44\n"

      : [inptr0] "+r" (inptr0),
        [inptr1] "+r" (inptr1),
        [inptr2] "+r" (inptr2),
        [outptr0] "+r" (outptr0),
        [outptr4] "+r" (outptr4),
        [outptr8] "+r" (outptr8),
        [outptr12] "+r" (outptr12),
        [n_remaining_channels] "+r" (n_remaining_channels)
      : [mstride1] "r" (sizeof(float) * mstride),
        [mstride2] "r" (sizeof(float) * mstride * 2),
        [mstride3] "r" (sizeof(float) * mstride * 3),
        [colstride1] "r" (sizeof(float) * kernel_col_stride),
        [colstride2] "r" (sizeof(float) * kernel_col_stride * 2),
        [one_half] "r" (0.5f)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24"
    );

    // Progression to complete stride
    outptr0 += matrix_row_stride - n_output_channels;
    outptr4 += matrix_row_stride - n_output_channels;
    outptr8 += matrix_row_stride - n_output_channels;
    outptr12 += matrix_row_stride - n_output_channels;
  }
}

template <>
template <>
inline void winograd2x2_3x3_gemm_kernel_transform_impl<float>::transform_kernel<1>(
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
    int n_remaining_channels = n_output_channels;

    asm volatile (
        // Registers into which to read the kernel
        "w_11 .req v0\n"  "qw_11 .req q0\n"  "sw_11 .req s0\n"
        "w_12 .req v1\n"  "qw_12 .req q1\n"  "sw_12 .req s1\n"
        "w_13 .req v2\n"  "qw_13 .req q2\n"  "sw_13 .req s2\n"
        "w_21 .req v3\n"  "qw_21 .req q3\n"  "sw_21 .req s3\n"
        "w_22 .req v4\n"  "qw_22 .req q4\n"  "sw_22 .req s4\n"
        "w_23 .req v5\n"  "qw_23 .req q5\n"  "sw_23 .req s5\n"
        "w_31 .req v6\n"  "qw_31 .req q6\n"  "sw_31 .req s6\n"
        "w_32 .req v7\n"  "qw_32 .req q7\n"  "sw_32 .req s7\n"
        "w_33 .req v8\n"  "qw_33 .req q8\n"  "sw_33 .req s8\n"

        // Transformed matrix Ww
        "Ww11 .req w_11\n"  "Ww12 .req w_12\n"  "Ww13 .req w_13\n"
        "Ww21 .req  v9\n"   "Ww22 .req v10\n"   "Ww23 .req v11\n"
        "Ww31 .req v12\n"   "Ww32 .req v13\n"   "Ww33 .req v14\n"
        "Ww41 .req w_31\n"  "Ww42 .req w_32\n"  "Ww43 .req w_33\n"

        // Output matrix U = WwWT
        "U11 .req Ww11\n"   "U12 .req v15\n"  "U13 .req v16\n"  "U14 .req Ww13\n"
        "U21 .req Ww21\n"   "U22 .req v17\n"  "U23 .req v18\n"  "U24 .req Ww23\n"
        "U31 .req Ww31\n"   "U32 .req v19\n"  "U33 .req v20\n"  "U34 .req Ww33\n"
        "U41 .req Ww41\n"   "U42 .req v21\n"  "U43 .req v22\n"  "U44 .req Ww43\n"

        // Storage view of output matrices
        "qU11 .req   q0\n"   "qU12 .req q15\n"  "qU13 .req q16\n"  "qU14 .req   q2\n"
        "qU21 .req   q9\n"   "qU22 .req q17\n"  "qU23 .req q18\n"  "qU24 .req  q11\n"
        "qU31 .req  q12\n"   "qU32 .req q19\n"  "qU33 .req q20\n"  "qU34 .req  q14\n"
        "qU41 .req   q6\n"   "qU42 .req q21\n"  "qU43 .req q22\n"  "qU44 .req   q8\n"

        "sU11 .req   s0\n"   "sU12 .req s15\n"  "sU13 .req s16\n"  "sU14 .req   s2\n"
        "sU21 .req   s9\n"   "sU22 .req s17\n"  "sU23 .req s18\n"  "sU24 .req  s11\n"
        "sU31 .req  s12\n"   "sU32 .req s19\n"  "sU33 .req s20\n"  "sU34 .req  s14\n"
        "sU41 .req   s6\n"   "sU42 .req s21\n"  "sU43 .req s22\n"  "sU44 .req   s8\n"

        "half .req v23\n"  // {0.5, ..., 0.5}
        "dup half.4s, %w[one_half]\n"
        "scratch .req v24\n"
        
        // Subtract the tail from the number of remaining channels and jump to
        // the tail if necessary.
        "subs %x[n_remaining_channels], %x[n_remaining_channels], #1\n"
        "beq 2f\n"

        "1:"
          // Load tile of the kernel
          "ldr qw_11, [%x[inptr0]]\n"
          "str qU11, [%x[outptr0]]\n"
          "ldr qw_12, [%x[inptr0], %x[colstride1]]\n"
          "ldr qw_13, [%x[inptr0], %x[colstride2]]\n"
          "str qU14, [%x[outptr0], %x[mstride3]]\n"
          "add %x[inptr0], %x[inptr0], #0x10\n"

          "ldr qw_21, [%x[inptr1]]\n"
          "ldr qw_22, [%x[inptr1], %x[colstride1]]\n"
          "ldr qw_23, [%x[inptr1], %x[colstride2]]\n"
          "add %x[inptr1], %x[inptr1], #0x10\n"

          "ldr qw_31, [%x[inptr2]]\n"
          "str qU41, [%x[outptr12]]\n"
          "ldr qw_32, [%x[inptr2], %x[colstride1]]\n"
          "ldr qw_33, [%x[inptr2], %x[colstride2]]\n"
          "str qU44, [%x[outptr12], %x[mstride3]]\n"
          "add %x[inptr2], %x[inptr2], #0x10\n"

          // Compute 2nd and 3rd rows of Ww
          "fadd scratch.4s, w_11.4s, w_31.4s\n"
          "fmul Ww21.4s, scratch.4s, half.4s\n"
          "fmla Ww21.4s, w_21.4s, half.4s\n"
          "str qU21, [%x[outptr4]]\n"
          "fmul Ww31.4s, scratch.4s, half.4s\n"
          "fmls Ww31.4s, w_21.4s, half.4s\n"
          "str qU31, [%x[outptr8]]\n"

          "fadd scratch.4s, w_12.4s, w_32.4s\n"
          "fmul Ww22.4s, scratch.4s, half.4s\n"
          "fmla Ww22.4s, w_22.4s, half.4s\n"
          "fmul Ww32.4s, scratch.4s, half.4s\n"
          "fmls Ww32.4s, w_22.4s, half.4s\n"

          "fadd scratch.4s, w_13.4s, w_33.4s\n"
          "fmul Ww23.4s, scratch.4s, half.4s\n"
          "fmla Ww23.4s, w_23.4s, half.4s\n"
          "str qU24, [%x[outptr4], %x[mstride3]]\n"
          "fmul Ww33.4s, scratch.4s, half.4s\n"
          "fmls Ww33.4s, w_23.4s, half.4s\n"
          "str qU34, [%x[outptr8], %x[mstride3]]\n"

          // Compute and store U, only need to compute the 2nd and 3rd columns
          // of U and update output pointers
          "fadd scratch.4s, Ww11.4s, Ww13.4s\n"
          "fmul U12.4s, scratch.4s, half.4s\n"
          "fmla U12.4s, Ww12.4s, half.4s\n"
          "str qU12, [%x[outptr0], %x[mstride1]]\n"
          "fmul U13.4s, scratch.4s, half.4s\n"
          "fmls U13.4s, Ww12.4s, half.4s\n"
          "str qU13, [%x[outptr0], %x[mstride2]]\n"
          "add  %x[outptr0],  %x[outptr0], #0x10\n"

          "fadd scratch.4s, Ww21.4s, Ww23.4s\n"
          "fmul U22.4s, scratch.4s, half.4s\n"
          "fmla U22.4s, Ww22.4s, half.4s\n"
          "str qU22, [%x[outptr4], %x[mstride1]]\n"
          "fmul U23.4s, scratch.4s, half.4s\n"
          "fmls U23.4s, Ww22.4s, half.4s\n"
          "str qU23, [%x[outptr4], %x[mstride2]]\n"
          "add  %x[outptr4],  %x[outptr4], #0x10\n"

          "fadd scratch.4s, Ww31.4s, Ww33.4s\n"
          "fmul U32.4s, scratch.4s, half.4s\n"
          "fmla U32.4s, Ww32.4s, half.4s\n"
          "str qU32, [%x[outptr8], %x[mstride1]]\n"
          "fmul U33.4s, scratch.4s, half.4s\n"
          "fmls U33.4s, Ww32.4s, half.4s\n"
          "str qU33, [%x[outptr8], %x[mstride2]]\n"
          "add  %x[outptr8],  %x[outptr8], #0x10\n"

          "fadd scratch.4s, Ww41.4s, Ww43.4s\n"
          "fmul U42.4s, scratch.4s, half.4s\n"
          "fmla U42.4s, Ww42.4s, half.4s\n"
          "str qU42, [%x[outptr12], %x[mstride1]]\n"
          "fmul U43.4s, scratch.4s, half.4s\n"
          "fmls U43.4s, Ww42.4s, half.4s\n"
          "str qU43, [%x[outptr12], %x[mstride2]]\n"
          "add %x[outptr12], %x[outptr12], #0x10\n"

          "subs %x[n_remaining_channels], %x[n_remaining_channels], #4\n"
          "bne 1b\n"

        // Tail size 1
        "2:"
          // Load tile of the kernel
          "ldr sw_11, [%x[inptr0]]\n"
          "str sU11, [%x[outptr0]]\n"
          "ldr sw_12, [%x[inptr0], %x[colstride1]]\n"
          "ldr sw_13, [%x[inptr0], %x[colstride2]]\n"
          "str sU14, [%x[outptr0], %x[mstride3]]\n"
          "add %x[inptr0], %x[inptr0], #0x04\n"

          "ldr sw_21, [%x[inptr1]]\n"
          "ldr sw_22, [%x[inptr1], %x[colstride1]]\n"
          "ldr sw_23, [%x[inptr1], %x[colstride2]]\n"
          "add %x[inptr1], %x[inptr1], #0x04\n"

          "ldr sw_31, [%x[inptr2]]\n"
          "str sU41, [%x[outptr12]]\n"
          "ldr sw_32, [%x[inptr2], %x[colstride1]]\n"
          "ldr sw_33, [%x[inptr2], %x[colstride2]]\n"
          "str sU44, [%x[outptr12], %x[mstride3]]\n"
          "add %x[inptr2], %x[inptr2], #0x04\n"

          // Compute 2nd and 3rd rows of Ww
          "fadd scratch.2s, w_11.2s, w_31.2s\n"
          "fmul Ww21.2s, scratch.2s, half.2s\n"
          "fmla Ww21.2s, w_21.2s, half.2s\n"
          "str sU21, [%x[outptr4]]\n"
          "fmul Ww31.2s, scratch.2s, half.2s\n"
          "fmls Ww31.2s, w_21.2s, half.2s\n"
          "str sU31, [%x[outptr8]]\n"

          "fadd scratch.2s, w_12.2s, w_32.2s\n"
          "fmul Ww22.2s, scratch.2s, half.2s\n"
          "fmla Ww22.2s, w_22.2s, half.2s\n"
          "fmul Ww32.2s, scratch.2s, half.2s\n"
          "fmls Ww32.2s, w_22.2s, half.2s\n"

          "fadd scratch.2s, w_13.2s, w_33.2s\n"
          "fmul Ww23.2s, scratch.2s, half.2s\n"
          "fmla Ww23.2s, w_23.2s, half.2s\n"
          "str sU24, [%x[outptr4], %x[mstride3]]\n"
          "fmul Ww33.2s, scratch.2s, half.2s\n"
          "fmls Ww33.2s, w_23.2s, half.2s\n"
          "str sU34, [%x[outptr8], %x[mstride3]]\n"

          // Compute and store U, only need to compute the 2nd and 3rd columns of
          // U and update output pointers
          "fadd scratch.2s, Ww11.2s, Ww13.2s\n"
          "fmul U12.2s, scratch.2s, half.2s\n"
          "fmla U12.2s, Ww12.2s, half.2s\n"
          "str sU12, [%x[outptr0], %x[mstride1]]\n"
          "fmul U13.2s, scratch.2s, half.2s\n"
          "fmls U13.2s, Ww12.2s, half.2s\n"
          "str sU13, [%x[outptr0], %x[mstride2]]\n"
          "add  %x[outptr0],  %x[outptr0], #0x04\n"

          "fadd scratch.2s, Ww21.2s, Ww23.2s\n"
          "fmul U22.2s, scratch.2s, half.2s\n"
          "fmla U22.2s, Ww22.2s, half.2s\n"
          "str sU22, [%x[outptr4], %x[mstride1]]\n"
          "fmul U23.2s, scratch.2s, half.2s\n"
          "fmls U23.2s, Ww22.2s, half.2s\n"
          "str sU23, [%x[outptr4], %x[mstride2]]\n"
          "add  %x[outptr4],  %x[outptr4], #0x04\n"

          "fadd scratch.2s, Ww31.2s, Ww33.2s\n"
          "fmul U32.2s, scratch.2s, half.2s\n"
          "fmla U32.2s, Ww32.2s, half.2s\n"
          "str sU32, [%x[outptr8], %x[mstride1]]\n"
          "fmul U33.2s, scratch.2s, half.2s\n"
          "fmls U33.2s, Ww32.2s, half.2s\n"
          "str sU33, [%x[outptr8], %x[mstride2]]\n"
          "add  %x[outptr8],  %x[outptr8], #0x04\n"

          "fadd scratch.2s, Ww41.2s, Ww43.2s\n"
          "fmul U42.2s, scratch.2s, half.2s\n"
          "fmla U42.2s, Ww42.2s, half.2s\n"
          "str sU42, [%x[outptr12], %x[mstride1]]\n"
          "fmul U43.2s, scratch.2s, half.2s\n"
          "fmls U43.2s, Ww42.2s, half.2s\n"
          "str sU43, [%x[outptr12], %x[mstride2]]\n"
          "add %x[outptr12], %x[outptr12], #0x04\n"

        // Clear aliases
        ".unreq half\n"
        ".unreq scratch\n"
        ".unreq w_11\n"  ".unreq qw_11\n" ".unreq sw_11\n"
        ".unreq w_12\n"  ".unreq qw_12\n" ".unreq sw_12\n"
        ".unreq w_13\n"  ".unreq qw_13\n" ".unreq sw_13\n"
        ".unreq w_21\n"  ".unreq qw_21\n" ".unreq sw_21\n"
        ".unreq w_22\n"  ".unreq qw_22\n" ".unreq sw_22\n"
        ".unreq w_23\n"  ".unreq qw_23\n" ".unreq sw_23\n"
        ".unreq w_31\n"  ".unreq qw_31\n" ".unreq sw_31\n"
        ".unreq w_32\n"  ".unreq qw_32\n" ".unreq sw_32\n"
        ".unreq w_33\n"  ".unreq qw_33\n" ".unreq sw_33\n"
        ".unreq Ww11\n"  ".unreq Ww12\n"  ".unreq Ww13\n"
        ".unreq Ww21\n"  ".unreq Ww22\n"  ".unreq Ww23\n"
        ".unreq Ww31\n"  ".unreq Ww32\n"  ".unreq Ww33\n"
        ".unreq Ww41\n"  ".unreq Ww42\n"  ".unreq Ww43\n"
        ".unreq U11\n"   ".unreq U12\n"   ".unreq U13\n"   ".unreq U14\n"
        ".unreq U21\n"   ".unreq U22\n"   ".unreq U23\n"   ".unreq U24\n"
        ".unreq U31\n"   ".unreq U32\n"   ".unreq U33\n"   ".unreq U34\n"
        ".unreq U41\n"   ".unreq U42\n"   ".unreq U43\n"   ".unreq U44\n"
        ".unreq qU11\n"  ".unreq qU12\n"  ".unreq qU13\n"  ".unreq qU14\n"
        ".unreq qU21\n"  ".unreq qU22\n"  ".unreq qU23\n"  ".unreq qU24\n"
        ".unreq qU31\n"  ".unreq qU32\n"  ".unreq qU33\n"  ".unreq qU34\n"
        ".unreq qU41\n"  ".unreq qU42\n"  ".unreq qU43\n"  ".unreq qU44\n"
        ".unreq sU11\n"  ".unreq sU12\n"  ".unreq sU13\n"  ".unreq sU14\n"
        ".unreq sU21\n"  ".unreq sU22\n"  ".unreq sU23\n"  ".unreq sU24\n"
        ".unreq sU31\n"  ".unreq sU32\n"  ".unreq sU33\n"  ".unreq sU34\n"
        ".unreq sU41\n"  ".unreq sU42\n"  ".unreq sU43\n"  ".unreq sU44\n"

      : [inptr0] "+r" (inptr0),
        [inptr1] "+r" (inptr1),
        [inptr2] "+r" (inptr2),
        [outptr0] "+r" (outptr0),
        [outptr4] "+r" (outptr4),
        [outptr8] "+r" (outptr8),
        [outptr12] "+r" (outptr12),
        [n_remaining_channels] "+r" (n_remaining_channels)
      : [mstride1] "r" (sizeof(float) * mstride),
        [mstride2] "r" (sizeof(float) * mstride * 2),
        [mstride3] "r" (sizeof(float) * mstride * 3),
        [colstride1] "r" (sizeof(float) * kernel_col_stride),
        [colstride2] "r" (sizeof(float) * kernel_col_stride * 2),
        [one_half] "r" (0.5f)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24"
    );

    // Progression to complete stride
    outptr0 += matrix_row_stride - n_output_channels;
    outptr4 += matrix_row_stride - n_output_channels;
    outptr8 += matrix_row_stride - n_output_channels;
    outptr12 += matrix_row_stride - n_output_channels;
  }
}
}
#endif  // __aarch64__
