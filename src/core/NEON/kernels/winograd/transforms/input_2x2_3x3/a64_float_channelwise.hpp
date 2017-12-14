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
#include "../input_2x2_3x3.hpp"

#ifdef __aarch64__

namespace winograd {

template <>
template <>
inline void Winograd2x2_3x3GemmInputChannelwise<float>::_process_tile<0, 0, 0, 0, 4>(
    int &n_channels,  // Number of channels in the tile
    const float* &inptr0,
    const int input_row_stride,
    const int input_col_stride,
    float* &outptr0,
    const int matrix_stride
) {
  // We use 4 pointers to point to the starting position on each row and use
  // three offsets to extract elements from each of the other 3 columns.
  auto inptr1 = inptr0 + 1*input_row_stride;
  auto inptr2 = inptr0 + 2*input_row_stride;
  auto inptr3 = inptr0 + 3*input_row_stride;

  // We use 4 pointers to point at matrices 0, 4, 8 and 12 and use three
  // offsets to access the intermediate matrices.
  auto outptr1 = outptr0 + matrix_stride * 4;
  auto outptr2 = outptr0 + matrix_stride * 8;
  auto outptr3 = outptr0 + matrix_stride * 12;

  for (; n_channels > 3; n_channels -= 4) {
    asm volatile (
        "X_11 .req  v0\n"  "qX_11 .req  q0\n"
        "X_12 .req  v1\n"  "qX_12 .req  q1\n"
        "X_13 .req  v2\n"  "qX_13 .req  q2\n"
        "X_14 .req  v3\n"  "qX_14 .req  q3\n"
        "X_21 .req  v4\n"  "qX_21 .req  q4\n"
        "X_22 .req  v5\n"  "qX_22 .req  q5\n"
        "X_23 .req  v6\n"  "qX_23 .req  q6\n"
        "X_24 .req  v7\n"  "qX_24 .req  q7\n"
        "X_31 .req  v8\n"  "qX_31 .req  q8\n"
        "X_32 .req  v9\n"  "qX_32 .req  q9\n"
        "X_33 .req v10\n"  "qX_33 .req q10\n"
        "X_34 .req v11\n"  "qX_34 .req q11\n"
        "X_41 .req v12\n"  "qX_41 .req q12\n"
        "X_42 .req v13\n"  "qX_42 .req q13\n"
        "X_43 .req v14\n"  "qX_43 .req q14\n"
        "X_44 .req v15\n"  "qX_44 .req q15\n"
        "xX_11 .req v16\n"
        "xX_12 .req v17\n"
        "xX_13 .req v18\n"
        "xX_14 .req v19\n"
        "xX_21 .req v20\n"
        "xX_22 .req v21\n"
        "xX_23 .req v22\n"
        "xX_24 .req v23\n"
        "xX_31 .req v24\n"
        "xX_32 .req v25\n"
        "xX_33 .req v26\n"
        "xX_34 .req v27\n"
        "xX_41 .req v28\n"
        "xX_42 .req v29\n"
        "xX_43 .req v30\n"
        "xX_44 .req v31\n"
        " U .req v0\n"
        "qU .req q0\n"

        // Load the tile, and compute compute the matrix xX
        "ldr qX_11, [%x[inptr0]]\n"
        "ldr qX_12, [%x[inptr0], %x[colstride1]]\n"
        "ldr qX_13, [%x[inptr0], %x[colstride2]]\n"
        "ldr qX_14, [%x[inptr0], %x[colstride3]]\n"
        "add %x[inptr0], %x[inptr0], #0x10\n"

        "ldr qX_21, [%x[inptr1]]\n"
        "fsub xX_11.4s, x_11.4s, x_13.4s\n"
        "ldr qX_22, [%x[inptr1], %x[colstride1]]\n"
        "fadd xX_12.4s, x_12.4s, x_13.4s\n"
        "ldr qX_23, [%x[inptr1], %x[colstride2]]\n"
        "fsub xX_13.4s, x_13.4s, x_12.4s\n"
        "ldr qX_24, [%x[inptr1], %x[colstride3]]\n"
        "fsub xX_14.4s, x_12.4s, x_14.4s\n"
        "add %x[inptr1], %x[inptr1], #0x10\n"

        "ldr qX_31, [%x[inptr2]]\n"
        "fsub xX_21.4s, x_21.4s, x_23.4s\n"
        "ldr qX_32, [%x[inptr2], %x[colstride1]]\n"
        "fadd xX_22.4s, x_22.4s, x_23.4s\n"
        "ldr qX_33, [%x[inptr2], %x[colstride2]]\n"
        "fsub xX_23.4s, x_23.4s, x_22.4s\n"
        "ldr qX_34, [%x[inptr2], %x[colstride3]]\n"
        "fsub xX_24.4s, x_22.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], #0x10\n"

        "ldr qX_41, [%x[inptr3]]\n"
        "fsub xX_31.4s, x_31.4s, x_33.4s\n"
        "ldr qX_42, [%x[inptr3], %x[colstride1]]\n"
        "fadd xX_32.4s, x_32.4s, x_33.4s\n"
        "ldr qX_43, [%x[inptr3], %x[colstride2]]\n"
        "fsub xX_33.4s, x_33.4s, x_32.4s\n"
        "ldr qX_44, [%x[inptr3], %x[colstride3]]\n"
        "fsub xX_34.4s, x_32.4s, x_34.4s\n"
        "add %x[inptr3], %x[inptr3], #0x10\n"

        // Complete computing xX while beginning to compute and store
        // $U = X.T x X$

        "fsub xX_41.4s, x_41.4s, x_43.4s\n"

        "fsub U.4s, xX_11.4s, xX_31.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fsub U.4s, xX_12.4s, xX_32.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, xX_13.4s, xX_33.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, xX_14.4s, xX_34.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], #0x10\n"

        "fadd xX_42.4s, x_42.4s, x_43.4s\n"

        "fadd U.4s, xX_21.4s, xX_31.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, xX_22.4s, xX_32.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fadd U.4s, xX_23.4s, xX_33.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fadd U.4s, xX_24.4s, xX_34.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], #0x10\n"

        "fsub xX_43.4s, x_43.4s, x_42.4s\n"

        "fsub U.4s, xX_31.4s, xX_21.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fsub U.4s, xX_32.4s, xX_22.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, xX_33.4s, xX_23.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, xX_34.4s, xX_24.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], #0x10\n"

        "fsub xX_44.4s, x_42.4s, x_44.4s\n"

        "fsub U.4s, xX_21.4s, xX_41.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fsub U.4s, xX_22.4s, xX_42.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, xX_23.4s, xX_43.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "fsub U.4s, xX_24.4s, xX_44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "add %x[outptr12], %x[outptr12], #0x10\n"

        ".unreq qU\n"
        ".unreq U\n"
        ".unreq X_11\n"  ".unreq qX_11\n"
        ".unreq X_12\n"  ".unreq qX_12\n"
        ".unreq X_13\n"  ".unreq qX_13\n"
        ".unreq X_14\n"  ".unreq qX_14\n"
        ".unreq X_21\n"  ".unreq qX_21\n"
        ".unreq X_22\n"  ".unreq qX_22\n"
        ".unreq X_23\n"  ".unreq qX_23\n"
        ".unreq X_24\n"  ".unreq qX_24\n"
        ".unreq X_31\n"  ".unreq qX_31\n"
        ".unreq X_32\n"  ".unreq qX_32\n"
        ".unreq X_33\n"  ".unreq qX_33\n"
        ".unreq X_34\n"  ".unreq qX_34\n"
        ".unreq X_41\n"  ".unreq qX_41\n"
        ".unreq X_42\n"  ".unreq qX_42\n"
        ".unreq X_43\n"  ".unreq qX_43\n"
        ".unreq X_44\n"  ".unreq qX_44\n"
        ".unreq xX_11\n"
        ".unreq xX_12\n"
        ".unreq xX_13\n"
        ".unreq xX_14\n"
        ".unreq xX_21\n"
        ".unreq xX_22\n"
        ".unreq xX_23\n"
        ".unreq xX_24\n"
        ".unreq xX_31\n"
        ".unreq xX_32\n"
        ".unreq xX_33\n"
        ".unreq xX_34\n"
        ".unreq xX_41\n"
        ".unreq xX_42\n"
        ".unreq xX_43\n"
        ".unreq xX_44\n"

        : [inptr0] "+r" (inptr0),
          [inptr1] "+r" (inptr1),
          [inptr2] "+r" (inptr2),
          [inptr3] "+r" (inptr3),
          [outptr0] "+r" (outptr0),
          [outptr4] "+r" (outptr1),
          [outptr8] "+r" (outptr2),
          [outptr12] "+r" (outptr3)
        : [colstride1] "r" (input_col_stride * sizeof(float)),
          [colstride2] "r" (input_col_stride * sizeof(float) * 2),
          [colstride3] "r" (input_col_stride * sizeof(float) * 3),
          [mstride1] "r" (matrix_stride * sizeof(float)),
          [mstride2] "r" (matrix_stride * sizeof(float) * 2),
          [mstride3] "r" (matrix_stride * sizeof(float) * 3)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31"
    );
  }
}

// Pad top by 1
template <>
template <>
inline void Winograd2x2_3x3GemmInputChannelwise<float>::_process_tile<1, 0, 0, 0, 4>(
    int &n_channels,  // Number of channels in the tile
    const float* &inptr0,
    const int input_row_stride,
    const int input_col_stride,
    float* &outptr0,
    const int matrix_stride
) {
  // We use 4 pointers to point to the starting position on each row and use
  // three offsets to extract elements from each of the other 3 columns.
  auto inptr1 = inptr0 + 0*input_row_stride;
  auto inptr2 = inptr0 + 1*input_row_stride;

  // We use 4 pointers to point at matrices 0, 4, 8 and 12 and use three
  // offsets to access the intermediate matrices.
  auto outptr1 = outptr0 + matrix_stride * 4;
  auto outptr2 = outptr0 + matrix_stride * 8;
  auto outptr3 = outptr0 + matrix_stride * 12;

  for (; n_channels > 3; n_channels -= 4) {
    asm volatile (
        "X_21 .req  v4\n"  "qX_21 .req  q4\n"
        "X_22 .req  v5\n"  "qX_22 .req  q5\n"
        "X_23 .req  v6\n"  "qX_23 .req  q6\n"
        "X_24 .req  v7\n"  "qX_24 .req  q7\n"
        "X_31 .req  v8\n"  "qX_31 .req  q8\n"
        "X_32 .req  v9\n"  "qX_32 .req  q9\n"
        "X_33 .req v10\n"  "qX_33 .req q10\n"
        "X_34 .req v11\n"  "qX_34 .req q11\n"
        "X_41 .req v12\n"  "qX_41 .req q12\n"
        "X_42 .req v13\n"  "qX_42 .req q13\n"
        "X_43 .req v14\n"  "qX_43 .req q14\n"
        "X_44 .req v15\n"  "qX_44 .req q15\n"
        "xX_21 .req v20\n"
        "xX_22 .req v21\n"
        "xX_23 .req v22\n"
        "xX_24 .req v23\n"
        "xX_31 .req v24\n"
        "xX_32 .req v25\n"
        "xX_33 .req v26\n"
        "xX_34 .req v27\n"
        "xX_41 .req v28\n"
        "xX_42 .req v29\n"
        "xX_43 .req v30\n"
        "xX_44 .req v31\n"
        " U .req v0\n"
        "qU .req q0\n"

        // Load the tile, and compute compute the matrix xX
        "ldr qX_21, [%x[inptr1]]\n"
        "ldr qX_22, [%x[inptr1], %x[colstride1]]\n"
        "ldr qX_23, [%x[inptr1], %x[colstride2]]\n"
        "ldr qX_24, [%x[inptr1], %x[colstride3]]\n"
        "add %x[inptr1], %x[inptr1], #0x10\n"

        "ldr qX_31, [%x[inptr2]]\n"
        "fsub xX_21.4s, x_21.4s, x_23.4s\n"
        "ldr qX_32, [%x[inptr2], %x[colstride1]]\n"
        "fadd xX_22.4s, x_22.4s, x_23.4s\n"
        "ldr qX_33, [%x[inptr2], %x[colstride2]]\n"
        "fsub xX_23.4s, x_23.4s, x_22.4s\n"
        "ldr qX_34, [%x[inptr2], %x[colstride3]]\n"
        "fsub xX_24.4s, x_22.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], #0x10\n"

        "ldr qX_41, [%x[inptr3]]\n"
        "fsub xX_31.4s, x_31.4s, x_33.4s\n"
        "ldr qX_42, [%x[inptr3], %x[colstride1]]\n"
        "fadd xX_32.4s, x_32.4s, x_33.4s\n"
        "ldr qX_43, [%x[inptr3], %x[colstride2]]\n"
        "fsub xX_33.4s, x_33.4s, x_32.4s\n"
        "ldr qX_44, [%x[inptr3], %x[colstride3]]\n"
        "fsub xX_34.4s, x_32.4s, x_34.4s\n"
        "add %x[inptr3], %x[inptr3], #0x10\n"

        // Complete computing xX while beginning to compute and store
        // $U = X.T x X$

        "fsub xX_41.4s, x_41.4s, x_43.4s\n"

        "fneg U.4s, xX_31.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fneg U.4s, xX_32.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fneg U.4s, xX_33.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fneg U.4s, xX_34.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], #0x10\n"

        "fadd xX_42.4s, x_42.4s, x_43.4s\n"

        "fadd U.4s, xX_21.4s, xX_31.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, xX_22.4s, xX_32.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fadd U.4s, xX_23.4s, xX_33.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fadd U.4s, xX_24.4s, xX_34.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], #0x10\n"

        "fsub xX_43.4s, x_43.4s, x_42.4s\n"

        "fsub U.4s, xX_31.4s, xX_21.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fsub U.4s, xX_32.4s, xX_22.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, xX_33.4s, xX_23.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, xX_34.4s, xX_24.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], #0x10\n"

        "fsub xX_44.4s, x_42.4s, x_44.4s\n"

        "fsub U.4s, xX_21.4s, xX_41.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fsub U.4s, xX_22.4s, xX_42.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, xX_23.4s, xX_43.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "fsub U.4s, xX_24.4s, xX_44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "add %x[outptr12], %x[outptr12], #0x10\n"

        ".unreq qU\n"
        ".unreq U\n"
        ".unreq X_21\n"  ".unreq qX_21\n"
        ".unreq X_22\n"  ".unreq qX_22\n"
        ".unreq X_23\n"  ".unreq qX_23\n"
        ".unreq X_24\n"  ".unreq qX_24\n"
        ".unreq X_31\n"  ".unreq qX_31\n"
        ".unreq X_32\n"  ".unreq qX_32\n"
        ".unreq X_33\n"  ".unreq qX_33\n"
        ".unreq X_34\n"  ".unreq qX_34\n"
        ".unreq X_41\n"  ".unreq qX_41\n"
        ".unreq X_42\n"  ".unreq qX_42\n"
        ".unreq X_43\n"  ".unreq qX_43\n"
        ".unreq X_44\n"  ".unreq qX_44\n"
        ".unreq xX_21\n"
        ".unreq xX_22\n"
        ".unreq xX_23\n"
        ".unreq xX_24\n"
        ".unreq xX_31\n"
        ".unreq xX_32\n"
        ".unreq xX_33\n"
        ".unreq xX_34\n"
        ".unreq xX_41\n"
        ".unreq xX_42\n"
        ".unreq xX_43\n"
        ".unreq xX_44\n"

        : [inptr1] "+r" (inptr0),  // Offset for missing row
          [inptr2] "+r" (inptr1),  // Offset for missing row
          [inptr3] "+r" (inptr2),  // Offset for missing row
          [outptr0] "+r" (outptr0),
          [outptr4] "+r" (outptr1),
          [outptr8] "+r" (outptr2),
          [outptr12] "+r" (outptr3)
        : [colstride1] "r" (input_col_stride * sizeof(float)),
          [colstride2] "r" (input_col_stride * sizeof(float) * 2),
          [colstride3] "r" (input_col_stride * sizeof(float) * 3),
          [mstride1] "r" (matrix_stride * sizeof(float)),
          [mstride2] "r" (matrix_stride * sizeof(float) * 2),
          [mstride3] "r" (matrix_stride * sizeof(float) * 3)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31"
    );
  }
}

// Pad left by 1
template <>
template <>
inline void Winograd2x2_3x3GemmInputChannelwise<float>::_process_tile<0, 1, 0, 0, 4>(
    int &n_channels,  // Number of channels in the tile
    const float* &inptr0,
    const int input_row_stride,
    const int input_col_stride,
    float* &outptr0,
    const int matrix_stride
) {
  // We use 4 pointers to point to the starting position on each row and use
  // three offsets to extract elements from each of the other 3 columns.
  auto inptr1 = inptr0 + 1*input_row_stride;
  auto inptr2 = inptr0 + 2*input_row_stride;
  auto inptr3 = inptr0 + 3*input_row_stride;

  // We use 4 pointers to point at matrices 0, 4, 8 and 12 and use three
  // offsets to access the intermediate matrices.
  auto outptr1 = outptr0 + matrix_stride * 4;
  auto outptr2 = outptr0 + matrix_stride * 8;
  auto outptr3 = outptr0 + matrix_stride * 12;

  for (; n_channels > 3; n_channels -= 4) {
    asm volatile (
        "X_12 .req  v1\n"  "qX_12 .req  q1\n"
        "X_13 .req  v2\n"  "qX_13 .req  q2\n"
        "X_14 .req  v3\n"  "qX_14 .req  q3\n"
        "X_22 .req  v5\n"  "qX_22 .req  q5\n"
        "X_23 .req  v6\n"  "qX_23 .req  q6\n"
        "X_24 .req  v7\n"  "qX_24 .req  q7\n"
        "X_32 .req  v9\n"  "qX_32 .req  q9\n"
        "X_33 .req v10\n"  "qX_33 .req q10\n"
        "X_34 .req v11\n"  "qX_34 .req q11\n"
        "X_42 .req v13\n"  "qX_42 .req q13\n"
        "X_43 .req v14\n"  "qX_43 .req q14\n"
        "X_44 .req v15\n"  "qX_44 .req q15\n"
        "xX_11 .req v16\n"
        "xX_12 .req v17\n"
        "xX_13 .req v18\n"
        "xX_14 .req v19\n"
        "xX_21 .req v20\n"
        "xX_22 .req v21\n"
        "xX_23 .req v22\n"
        "xX_24 .req v23\n"
        "xX_31 .req v24\n"
        "xX_32 .req v25\n"
        "xX_33 .req v26\n"
        "xX_34 .req v27\n"
        "xX_41 .req v28\n"
        "xX_42 .req v29\n"
        "xX_43 .req v30\n"
        "xX_44 .req v31\n"
        " U .req v0\n"
        "qU .req q0\n"

        // Load the tile, and compute compute the matrix xX
        "ldr qX_12, [%x[inptr0]]\n"
        "ldr qX_13, [%x[inptr0], %x[colstride1]]\n"
        "ldr qX_14, [%x[inptr0], %x[colstride2]]\n"
        "add %x[inptr0], %x[inptr0], #0x10\n"

        "fneg xX_11.4s, x_13.4s\n"
        "ldr qX_22, [%x[inptr1]]\n"
        "fadd xX_12.4s, x_12.4s, x_13.4s\n"
        "ldr qX_23, [%x[inptr1], %x[colstride1]]\n"
        "fsub xX_13.4s, x_13.4s, x_12.4s\n"
        "ldr qX_24, [%x[inptr1], %x[colstride2]]\n"
        "fsub xX_14.4s, x_12.4s, x_14.4s\n"
        "add %x[inptr1], %x[inptr1], #0x10\n"

        "fneg xX_21.4s, x_23.4s\n"
        "ldr qX_32, [%x[inptr2]]\n"
        "fadd xX_22.4s, x_22.4s, x_23.4s\n"
        "ldr qX_33, [%x[inptr2], %x[colstride1]]\n"
        "fsub xX_23.4s, x_23.4s, x_22.4s\n"
        "ldr qX_34, [%x[inptr2], %x[colstride2]]\n"
        "fsub xX_24.4s, x_22.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], #0x10\n"

        "fneg xX_31.4s, x_33.4s\n"
        "ldr qX_42, [%x[inptr3]]\n"
        "fadd xX_32.4s, x_32.4s, x_33.4s\n"
        "ldr qX_43, [%x[inptr3], %x[colstride1]]\n"
        "fsub xX_33.4s, x_33.4s, x_32.4s\n"
        "ldr qX_44, [%x[inptr3], %x[colstride2]]\n"
        "fsub xX_34.4s, x_32.4s, x_34.4s\n"
        "add %x[inptr3], %x[inptr3], #0x10\n"

        // Complete computing xX while beginning to compute and store
        // $U = X.T x X$

        "fneg xX_41.4s, x_43.4s\n"

        "fsub U.4s, xX_11.4s, xX_31.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fsub U.4s, xX_12.4s, xX_32.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, xX_13.4s, xX_33.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, xX_14.4s, xX_34.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], #0x10\n"

        "fadd xX_42.4s, x_42.4s, x_43.4s\n"

        "fadd U.4s, xX_21.4s, xX_31.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, xX_22.4s, xX_32.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fadd U.4s, xX_23.4s, xX_33.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fadd U.4s, xX_24.4s, xX_34.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], #0x10\n"

        "fsub xX_43.4s, x_43.4s, x_42.4s\n"

        "fsub U.4s, xX_31.4s, xX_21.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fsub U.4s, xX_32.4s, xX_22.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, xX_33.4s, xX_23.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, xX_34.4s, xX_24.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], #0x10\n"

        "fsub xX_44.4s, x_42.4s, x_44.4s\n"

        "fsub U.4s, xX_21.4s, xX_41.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fsub U.4s, xX_22.4s, xX_42.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, xX_23.4s, xX_43.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "fsub U.4s, xX_24.4s, xX_44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "add %x[outptr12], %x[outptr12], #0x10\n"

        ".unreq X_12\n"  ".unreq qX_12\n"
        ".unreq X_13\n"  ".unreq qX_13\n"
        ".unreq X_14\n"  ".unreq qX_14\n"
        ".unreq X_22\n"  ".unreq qX_22\n"
        ".unreq X_23\n"  ".unreq qX_23\n"
        ".unreq X_24\n"  ".unreq qX_24\n"
        ".unreq X_32\n"  ".unreq qX_32\n"
        ".unreq X_33\n"  ".unreq qX_33\n"
        ".unreq X_34\n"  ".unreq qX_34\n"
        ".unreq X_42\n"  ".unreq qX_42\n"
        ".unreq X_43\n"  ".unreq qX_43\n"
        ".unreq X_44\n"  ".unreq qX_44\n"
        ".unreq xX_11\n"
        ".unreq xX_12\n"
        ".unreq xX_13\n"
        ".unreq xX_14\n"
        ".unreq xX_21\n"
        ".unreq xX_22\n"
        ".unreq xX_23\n"
        ".unreq xX_24\n"
        ".unreq xX_31\n"
        ".unreq xX_32\n"
        ".unreq xX_33\n"
        ".unreq xX_34\n"
        ".unreq xX_41\n"
        ".unreq xX_42\n"
        ".unreq xX_43\n"
        ".unreq xX_44\n"
        ".unreq U\n"
        ".unreq qU\n"

        : [inptr0] "+r" (inptr0),
          [inptr1] "+r" (inptr1),
          [inptr2] "+r" (inptr2),
          [inptr3] "+r" (inptr3),
          [outptr0] "+r" (outptr0),
          [outptr4] "+r" (outptr1),
          [outptr8] "+r" (outptr2),
          [outptr12] "+r" (outptr3)
        : [colstride1] "r" (input_col_stride * sizeof(float)),
          [colstride2] "r" (input_col_stride * sizeof(float) * 2),
          [mstride1] "r" (matrix_stride * sizeof(float)),
          [mstride2] "r" (matrix_stride * sizeof(float) * 2),
          [mstride3] "r" (matrix_stride * sizeof(float) * 3)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31"
    );
  }
}

// Pad bottom by 1
template <>
template <>
inline void Winograd2x2_3x3GemmInputChannelwise<float>::_process_tile<0, 0, 1, 0, 4>(
    int &n_channels,  // Number of channels in the tile
    const float* &inptr0,
    const int input_row_stride,
    const int input_col_stride,
    float* &outptr0,
    const int matrix_stride
) {
  // We use 4 pointers to point to the starting position on each row and use
  // three offsets to extract elements from each of the other 3 columns.
  auto inptr1 = inptr0 + 1*input_row_stride;
  auto inptr2 = inptr0 + 2*input_row_stride;

  // We use 4 pointers to point at matrices 0, 4, 8 and 12 and use three
  // offsets to access the intermediate matrices.
  auto outptr1 = outptr0 + matrix_stride * 4;
  auto outptr2 = outptr0 + matrix_stride * 8;
  auto outptr3 = outptr0 + matrix_stride * 12;

  for (; n_channels > 3; n_channels -= 4) {
    asm volatile (
        "X_11 .req  v0\n"  "qX_11 .req  q0\n"
        "X_12 .req  v1\n"  "qX_12 .req  q1\n"
        "X_13 .req  v2\n"  "qX_13 .req  q2\n"
        "X_14 .req  v3\n"  "qX_14 .req  q3\n"
        "X_21 .req  v4\n"  "qX_21 .req  q4\n"
        "X_22 .req  v5\n"  "qX_22 .req  q5\n"
        "X_23 .req  v6\n"  "qX_23 .req  q6\n"
        "X_24 .req  v7\n"  "qX_24 .req  q7\n"
        "X_31 .req  v8\n"  "qX_31 .req  q8\n"
        "X_32 .req  v9\n"  "qX_32 .req  q9\n"
        "X_33 .req v10\n"  "qX_33 .req q10\n"
        "X_34 .req v11\n"  "qX_34 .req q11\n"
        "xX_11 .req v16\n"
        "xX_12 .req v17\n"
        "xX_13 .req v18\n"
        "xX_14 .req v19\n"
        "xX_21 .req v20\n" "qxX_21 .req q20\n"
        "xX_22 .req v21\n" "qxX_22 .req q21\n"
        "xX_23 .req v22\n" "qxX_23 .req q22\n"
        "xX_24 .req v23\n" "qxX_24 .req q23\n"
        "xX_31 .req v24\n"
        "xX_32 .req v25\n"
        "xX_33 .req v26\n"
        "xX_34 .req v27\n"
        " U .req v0\n"
        "qU .req q0\n"

        // Load the tile, and compute compute the matrix xX
        "ldr qX_11, [%x[inptr0]]\n"
        "ldr qX_12, [%x[inptr0], %x[colstride1]]\n"
        "ldr qX_13, [%x[inptr0], %x[colstride2]]\n"
        "ldr qX_14, [%x[inptr0], %x[colstride3]]\n"
        "add %x[inptr0], %x[inptr0], #0x10\n"

        "ldr qX_21, [%x[inptr1]]\n"
        "fsub xX_11.4s, x_11.4s, x_13.4s\n"
        "ldr qX_22, [%x[inptr1], %x[colstride1]]\n"
        "fadd xX_12.4s, x_12.4s, x_13.4s\n"
        "ldr qX_23, [%x[inptr1], %x[colstride2]]\n"
        "fsub xX_13.4s, x_13.4s, x_12.4s\n"
        "ldr qX_24, [%x[inptr1], %x[colstride3]]\n"
        "fsub xX_14.4s, x_12.4s, x_14.4s\n"
        "add %x[inptr1], %x[inptr1], #0x10\n"

        "ldr qX_31, [%x[inptr2]]\n"
        "fsub xX_21.4s, x_21.4s, x_23.4s\n"
        "ldr qX_32, [%x[inptr2], %x[colstride1]]\n"
        "fadd xX_22.4s, x_22.4s, x_23.4s\n"
        "ldr qX_33, [%x[inptr2], %x[colstride2]]\n"
        "fsub xX_23.4s, x_23.4s, x_22.4s\n"
        "ldr qX_34, [%x[inptr2], %x[colstride3]]\n"
        "fsub xX_24.4s, x_22.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], #0x10\n"

        "fsub xX_31.4s, x_31.4s, x_33.4s\n"
        "fadd xX_32.4s, x_32.4s, x_33.4s\n"
        "fsub xX_33.4s, x_33.4s, x_32.4s\n"
        "fsub xX_34.4s, x_32.4s, x_34.4s\n"

        // Complete computing xX while beginning to compute and store
        // $U = X.T x X$

        "fsub U.4s, xX_11.4s, xX_31.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fsub U.4s, xX_12.4s, xX_32.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, xX_13.4s, xX_33.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, xX_14.4s, xX_34.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], #0x10\n"

        "fadd U.4s, xX_21.4s, xX_31.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, xX_22.4s, xX_32.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fadd U.4s, xX_23.4s, xX_33.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fadd U.4s, xX_24.4s, xX_34.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], #0x10\n"

        "fsub U.4s, xX_31.4s, xX_21.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fsub U.4s, xX_32.4s, xX_22.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, xX_33.4s, xX_23.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, xX_34.4s, xX_24.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], #0x10\n"

        "str qxX_21, [%x[outptr12]]\n"
        "str qxX_22, [%x[outptr12], %x[mstride1]]\n"
        "str qxX_23, [%x[outptr12], %x[mstride2]]\n"
        "str qxX_24, [%x[outptr12], %x[mstride3]]\n"
        "add %x[outptr12], %x[outptr12], #0x10\n"

        ".unreq qU\n"
        ".unreq U\n"
        ".unreq X_11\n"  ".unreq qX_11\n"
        ".unreq X_12\n"  ".unreq qX_12\n"
        ".unreq X_13\n"  ".unreq qX_13\n"
        ".unreq X_14\n"  ".unreq qX_14\n"
        ".unreq X_21\n"  ".unreq qX_21\n"
        ".unreq X_22\n"  ".unreq qX_22\n"
        ".unreq X_23\n"  ".unreq qX_23\n"
        ".unreq X_24\n"  ".unreq qX_24\n"
        ".unreq X_31\n"  ".unreq qX_31\n"
        ".unreq X_32\n"  ".unreq qX_32\n"
        ".unreq X_33\n"  ".unreq qX_33\n"
        ".unreq X_34\n"  ".unreq qX_34\n"
        ".unreq xX_11\n"
        ".unreq xX_12\n"
        ".unreq xX_13\n"
        ".unreq xX_14\n"
        ".unreq xX_21\n" ".unreq qxX_21\n"
        ".unreq xX_22\n" ".unreq qxX_22\n"
        ".unreq xX_23\n" ".unreq qxX_23\n"
        ".unreq xX_24\n" ".unreq qxX_24\n"
        ".unreq xX_31\n"
        ".unreq xX_32\n"
        ".unreq xX_33\n"
        ".unreq xX_34\n"

        : [inptr0] "+r" (inptr0),
          [inptr1] "+r" (inptr1),
          [inptr2] "+r" (inptr2),
          [outptr0] "+r" (outptr0),
          [outptr4] "+r" (outptr1),
          [outptr8] "+r" (outptr2),
          [outptr12] "+r" (outptr3)
        : [colstride1] "r" (input_col_stride * sizeof(float)),
          [colstride2] "r" (input_col_stride * sizeof(float) * 2),
          [colstride3] "r" (input_col_stride * sizeof(float) * 3),
          [mstride1] "r" (matrix_stride * sizeof(float)),
          [mstride2] "r" (matrix_stride * sizeof(float) * 2),
          [mstride3] "r" (matrix_stride * sizeof(float) * 3)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31"
    );
  }
}

// Pad right by 1
template <>
template <>
inline void Winograd2x2_3x3GemmInputChannelwise<float>::_process_tile<0, 0, 0, 1, 4>(
    int &n_channels,  // Number of channels in the tile
    const float* &inptr0,
    const int input_row_stride,
    const int input_col_stride,
    float* &outptr0,
    const int matrix_stride
) {
  // We use 4 pointers to point to the starting position on each row and use
  // three offsets to extract elements from each of the other 3 columns.
  auto inptr1 = inptr0 + 1*input_row_stride;
  auto inptr2 = inptr0 + 2*input_row_stride;
  auto inptr3 = inptr0 + 3*input_row_stride;

  // We use 4 pointers to point at matrices 0, 4, 8 and 12 and use three
  // offsets to access the intermediate matrices.
  auto outptr1 = outptr0 + matrix_stride * 4;
  auto outptr2 = outptr0 + matrix_stride * 8;
  auto outptr3 = outptr0 + matrix_stride * 12;

  for (; n_channels > 3; n_channels -= 4) {
    asm volatile (
        "X_11 .req  v0\n"  "qX_11 .req  q0\n"
        "X_12 .req  v1\n"  "qX_12 .req  q1\n"
        "X_13 .req  v2\n"  "qX_13 .req  q2\n"
        "X_21 .req  v4\n"  "qX_21 .req  q4\n"
        "X_22 .req  v5\n"  "qX_22 .req  q5\n"
        "X_23 .req  v6\n"  "qX_23 .req  q6\n"
        "X_31 .req  v8\n"  "qX_31 .req  q8\n"
        "X_32 .req  v9\n"  "qX_32 .req  q9\n"
        "X_33 .req v10\n"  "qX_33 .req q10\n"
        "X_41 .req v12\n"  "qX_41 .req q12\n"
        "X_42 .req v13\n"  "qX_42 .req q13\n"
        "X_43 .req v14\n"  "qX_43 .req q14\n"
        "xX_11 .req v16\n"
        "xX_12 .req v17\n"
        "xX_13 .req v18\n"
        "xX_14 .req x_12\n"
        "xX_21 .req v20\n"
        "xX_22 .req v21\n"
        "xX_23 .req v22\n"
        "xX_24 .req x_22\n"
        "xX_31 .req v24\n"
        "xX_32 .req v25\n"
        "xX_33 .req v26\n"
        "xX_34 .req x_32\n"
        "xX_41 .req v28\n"
        "xX_42 .req v29\n"
        "xX_43 .req v30\n"
        "xX_44 .req x_42\n"
        " U .req v0\n"
        "qU .req q0\n"

        // Load the tile, and compute compute the matrix xX
        "ldr qX_11, [%x[inptr0]]\n"
        "ldr qX_12, [%x[inptr0], %x[colstride1]]\n"
        "ldr qX_13, [%x[inptr0], %x[colstride2]]\n"
        "add %x[inptr0], %x[inptr0], #0x10\n"

        "ldr qX_21, [%x[inptr1]]\n"
        "fsub xX_11.4s, x_11.4s, x_13.4s\n"
        "ldr qX_22, [%x[inptr1], %x[colstride1]]\n"
        "fadd xX_12.4s, x_12.4s, x_13.4s\n"
        "ldr qX_23, [%x[inptr1], %x[colstride2]]\n"
        "fsub xX_13.4s, x_13.4s, x_12.4s\n"
        "add %x[inptr1], %x[inptr1], #0x10\n"

        "ldr qX_31, [%x[inptr2]]\n"
        "fsub xX_21.4s, x_21.4s, x_23.4s\n"
        "ldr qX_32, [%x[inptr2], %x[colstride1]]\n"
        "fadd xX_22.4s, x_22.4s, x_23.4s\n"
        "ldr qX_33, [%x[inptr2], %x[colstride2]]\n"
        "fsub xX_23.4s, x_23.4s, x_22.4s\n"
        "add %x[inptr2], %x[inptr2], #0x10\n"

        "ldr qX_41, [%x[inptr3]]\n"
        "fsub xX_31.4s, x_31.4s, x_33.4s\n"
        "ldr qX_42, [%x[inptr3], %x[colstride1]]\n"
        "fadd xX_32.4s, x_32.4s, x_33.4s\n"
        "ldr qX_43, [%x[inptr3], %x[colstride2]]\n"
        "fsub xX_33.4s, x_33.4s, x_32.4s\n"
        "add %x[inptr3], %x[inptr3], #0x10\n"

        // Complete computing xX while beginning to compute and store
        // $U = X.T x X$

        "fsub xX_41.4s, x_41.4s, x_43.4s\n"

        "fsub U.4s, xX_11.4s, xX_31.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fsub U.4s, xX_12.4s, xX_32.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, xX_13.4s, xX_33.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, xX_14.4s, xX_34.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], #0x10\n"

        "fadd xX_42.4s, x_42.4s, x_43.4s\n"

        "fadd U.4s, xX_21.4s, xX_31.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, xX_22.4s, xX_32.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fadd U.4s, xX_23.4s, xX_33.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fadd U.4s, xX_24.4s, xX_34.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], #0x10\n"

        "fsub xX_43.4s, x_43.4s, x_42.4s\n"

        "fsub U.4s, xX_31.4s, xX_21.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fsub U.4s, xX_32.4s, xX_22.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, xX_33.4s, xX_23.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, xX_34.4s, xX_24.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], #0x10\n"

        "fsub U.4s, xX_21.4s, xX_41.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fsub U.4s, xX_22.4s, xX_42.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, xX_23.4s, xX_43.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "fsub U.4s, xX_24.4s, xX_44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "add %x[outptr12], %x[outptr12], #0x10\n"

        ".unreq qU\n"
        ".unreq U\n"
        ".unreq X_11\n"  ".unreq qX_11\n"
        ".unreq X_12\n"  ".unreq qX_12\n"
        ".unreq X_13\n"  ".unreq qX_13\n"
        ".unreq X_21\n"  ".unreq qX_21\n"
        ".unreq X_22\n"  ".unreq qX_22\n"
        ".unreq X_23\n"  ".unreq qX_23\n"
        ".unreq X_31\n"  ".unreq qX_31\n"
        ".unreq X_32\n"  ".unreq qX_32\n"
        ".unreq X_33\n"  ".unreq qX_33\n"
        ".unreq X_41\n"  ".unreq qX_41\n"
        ".unreq X_42\n"  ".unreq qX_42\n"
        ".unreq X_43\n"  ".unreq qX_43\n"
        ".unreq xX_11\n"
        ".unreq xX_12\n"
        ".unreq xX_13\n"
        ".unreq xX_14\n"
        ".unreq xX_21\n"
        ".unreq xX_22\n"
        ".unreq xX_23\n"
        ".unreq xX_24\n"
        ".unreq xX_31\n"
        ".unreq xX_32\n"
        ".unreq xX_33\n"
        ".unreq xX_34\n"
        ".unreq xX_41\n"
        ".unreq xX_42\n"
        ".unreq xX_43\n"
        ".unreq xX_44\n"

        : [inptr0] "+r" (inptr0),
          [inptr1] "+r" (inptr1),
          [inptr2] "+r" (inptr2),
          [inptr3] "+r" (inptr3),
          [outptr0] "+r" (outptr0),
          [outptr4] "+r" (outptr1),
          [outptr8] "+r" (outptr2),
          [outptr12] "+r" (outptr3)
        : [colstride1] "r" (input_col_stride * sizeof(float)),
          [colstride2] "r" (input_col_stride * sizeof(float) * 2),
          [mstride1] "r" (matrix_stride * sizeof(float)),
          [mstride2] "r" (matrix_stride * sizeof(float) * 2),
          [mstride3] "r" (matrix_stride * sizeof(float) * 3)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31"
    );
  }
}
}
#endif
