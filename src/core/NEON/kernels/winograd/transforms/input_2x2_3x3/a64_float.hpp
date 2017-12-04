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

// Pad left by one column, pad right by one column, no upper or lower padding, 4 channels
template <>
template <>
inline void Winograd2x2_3x3GemmInput<float>::process_tile_row<0, 1, 0, 1, 4>(
    const int tile_N,            // Number of tiles in the row
    const float* const input,    // Base input pointer (appropriate to batch, channel and row)
    const int input_row_stride,  // Stride between rows of the input
    const int input_col_stride,  // Stride between columns of the input
    float* const matrix,         // 1st output matrix (appropriate to batch, channel and row)
    const int matrix_stride,     // Stride between matrices
    const int matrix_row_stride  // Stride between rows of the output matrix
) {
  /* SIMD register allocation
   * ========================
   *
   * In the following code we read 4x4 tiles of a matrix `x`, with which we
   * compute another matrix `X.T x` where:
   *
   *         /  1  0  0  0 \
   *     X = |  0  1 -1  1 |
   *         | -1  1  1  0 |
   *         \  0  0  0 -1 /
   *
   * Hence, `X.T` is a program which operates upon rows of the matrix `X`.
   * We subsequently compute and store the matrix `U = (X.T x) X`.
   *
   * Importantly, each iteration of the loop below loads a new matrix `x'`
   * where the final two columns of `x'` are the first two columns of the
   * previous `x`. That is:
   *
   *   x11  x12  x13  x14
   *   x21  x22  x23  x24
   *   x31  x32  x33  x34
   *   x41  x42  x43  x44
   *
   *            x'11 x'12 x'13 x'14
   *            x'21 x'22 x'23 x'24
   *            x'31 x'32 x'33 x'34
   *            x'41 x'42 x'43 x'44
   *
   * Consequently, while the first iteration of the below loop must load 16
   * values for `x`, the second need load only 8. *Furthermore*, since we noted
   * above that the operation `X.T x` was a program which operated upon *rows*
   * of the matrix `x` it follows that that the relation that `x'[i][1] =
   * x[i][3]` and `x'[i][2] = x[i][4]` applies also the matrices `X.T x'` and
   * `X.T x`. That is:
   *
   *   (X.T x)11  (X.T x)12  (X.T x)13  (X.T x)14
   *   (X.T x)21  (X.T x)22  (X.T x)23  (X.T x)24
   *   (X.T x)31  (X.T x)32  (X.T x)33  (X.T x)34
   *   (X.T x)41  (X.T x)42  (X.T x)43  (X.T x)44
   *
   *                        (X.T x')11 (X.T x')12 (X.T x')13 (X.T x')14
   *                        (X.T x')12 (X.T x')12 (X.T x')12 (X.T x')12
   *                        (X.T x')13 (X.T x')13 (X.T x')13 (X.T x')13
   *                        (X.T x')14 (X.T x')14 (X.T x')14 (X.T x')14
   *
   * Hence, as well as not needing to load new values for x'[i][1..2] it is
   * also unnecessary to recompute values for (X.T x')[i][1..2].
   *
   * Following this we break the registers into blocks `A` and `B` used by the
   * two stages of the unrolled loop. These registers named such that the
   * latter columns of `A` become the earlier columns of `B` and vice-versa:
   *
   *  AXTx11 AXTx12 > AXTx13 AXTx14 |
   *  AXTx21 AXTx22 > AXTx23 AXTx24 |
   *  AXTx31 AXTx32 > AXTx33 AXTx34 |
   *  AXTx41 AXTx42 > AXTx43 AXTx44 |
   *
   *  BXTx13 BXTx14 | BXTx11 BXTx12 >
   *  BXTx23 BXTx24 | BXTx21 BXTx22 >
   *  BXTx33 BXTx34 | BXTx31 BXTx32 >
   *  BXTx43 BXTx44 | BXTx41 BXTx42 >
   *
   * These 32 named registers require only 16 architectural registers. 1
   * additional architectural register is used as scratch space and 8
   * architectural registers are used to load in the values x[1..4][3,4].
   *
   * Input and output addressing
   * ===========================
   * TODO Description
   */
  const float *inptr0 = input;
  const float *inptr1 = input + input_row_stride;
  const float *inptr2 = input + input_row_stride * 2;
  const float *inptr3 = input + input_row_stride * 3;

  float *outptr0 = matrix;
  float *outptr4 = matrix + matrix_stride * 4;
  float *outptr8 = matrix + matrix_stride * 8;
  float *outptr12 = matrix + matrix_stride * 12;

  int tile_j = tile_N;  // Tiles to process

  asm volatile (
      // Named SIMD registers according to the policy given above
      // Registers into which to load the latter two columns of `x`
      "x_13 .req v0\n qx_13 .req q0\n" "x_14 .req v4\n qx_14 .req q4\n"
      "x_23 .req v1\n qx_23 .req q1\n" "x_24 .req v5\n qx_24 .req q5\n"
      "x_33 .req v2\n qx_33 .req q2\n" "x_34 .req v6\n qx_34 .req q6\n"
      "x_43 .req v3\n qx_43 .req q3\n" "x_44 .req v7\n qx_44 .req q7\n"

      // Registers for storing X.T x (both A and B halves)
      "AXTx11 .req  v8\n" "BXTx13 .req  v8\n"
      "AXTx12 .req  v9\n" "BXTx14 .req  v9\n" "qAXTx12 .req  q9\n"
      "AXTx21 .req v10\n" "BXTx23 .req v10\n"
      "AXTx22 .req v11\n" "BXTx24 .req v11\n" "qAXTx22 .req q11\n"
      "AXTx31 .req v12\n" "BXTx33 .req v12\n"
      "AXTx32 .req v13\n" "BXTx34 .req v13\n" "qAXTx32 .req q13\n"
      "AXTx41 .req v14\n" "BXTx43 .req v14\n"
      "AXTx42 .req v15\n" "BXTx44 .req v15\n" "qAXTx42 .req q15\n"
      "AXTx13 .req v16\n" "BXTx11 .req v16\n"
      "AXTx14 .req v17\n" "BXTx12 .req v17\n" "qBXTx12 .req q17\n"
      "AXTx23 .req v18\n" "BXTx21 .req v18\n"
      "AXTx24 .req v19\n" "BXTx22 .req v19\n" "qBXTx22 .req q19\n"
      "AXTx33 .req v20\n" "BXTx31 .req v20\n"
      "AXTx34 .req v21\n" "BXTx32 .req v21\n" "qBXTx32 .req q21\n"
      "AXTx43 .req v22\n" "BXTx41 .req v22\n"
      "AXTx44 .req v23\n" "BXTx42 .req v23\n" "qBXTx42 .req q23\n"

      // Result register (TODO Does using more registers yield better
      // performance)
      "U .req v24\n qU .req q24\n"

      // ----------------------------------------------------------------------
      // Head of loop
      //   Loads a complete 4x4 tile of x, computes X.T x, computes and stores
      //   `U = X.T x X`. Prepares for the 'A' half of the loop.
      //   NOTE: Since the first tile has the leftmost column padded we can
      //   skip 4 loads and 4 calculations for the matrix X.T x X.

      // Temporarily alias registers for computing the first (non-padded)
      // column of x.
      "x_12 .req v0\n qx_12 .req q0\n"
      "x_22 .req v1\n qx_22 .req q1\n"
      "x_32 .req v2\n qx_32 .req q2\n"
      "x_42 .req v3\n qx_42 .req q3\n"

      "ldr qx_12, [%x[inptr0]]\n"
      "ldr qx_22, [%x[inptr1]]\n"
      "ldr qx_32, [%x[inptr2]]\n"
      "ldr qx_42, [%x[inptr3]]\n"

      "fsub BXTx12.4s, x_12.4s, x_32.4s\n"
      "fadd BXTx22.4s, x_22.4s, x_32.4s\n"
      "fsub BXTx32.4s, x_32.4s, x_22.4s\n"
      "fsub BXTx42.4s, x_22.4s, x_42.4s\n"

      ".unreq x_12\n .unreq qx_12\n"
      ".unreq x_22\n .unreq qx_22\n"
      ".unreq x_32\n .unreq qx_32\n"
      ".unreq x_42\n .unreq qx_42\n"

      // Load and compute latter two columns of the first tile. Progress the
      // input pointers (by three columns so that the each points are the
      // second column of the next tile, that is, each points at the first
      // column which must be read for the next tile.
      "ldr qx_13, [%x[inptr0], %x[colstride1]]\n"
      "ldr qx_23, [%x[inptr1], %x[colstride1]]\n"
      "ldr qx_33, [%x[inptr2], %x[colstride1]]\n"
      "ldr qx_43, [%x[inptr3], %x[colstride1]]\n"

      "fsub BXTx13.4s, x_13.4s, x_33.4s\n"
      "ldr qx_14, [%x[inptr0], %x[colstride2]]\n"

      "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
      "ldr qx_24, [%x[inptr1], %x[colstride2]]\n"

      "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
      "ldr qx_34, [%x[inptr2], %x[colstride2]]\n"

      "fsub BXTx43.4s, x_23.4s, x_43.4s\n"
      "ldr qx_44, [%x[inptr3], %x[colstride2]]\n"

      "fsub BXTx14.4s, x_14.4s, x_34.4s\n"
      "add %x[inptr0],  %x[inptr0], %x[colstride3]\n"

      "fadd BXTx24.4s, x_24.4s, x_34.4s\n"
      "add %x[inptr1], %x[inptr1], %x[colstride3]\n"

      "fsub BXTx34.4s, x_34.4s, x_24.4s\n"
      "add %x[inptr2], %x[inptr2], %x[colstride3]\n"

      "fsub BXTx44.4s, x_24.4s, x_44.4s\n"
      "add %x[inptr3], %x[inptr3], %x[colstride3]\n"

      // Compute and store U for the first tile
      // First row
      "fneg U.4s, BXTx13.4s\n"
      "str qU, [%x[outptr0]]\n"
      "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
      "str qU, [%x[outptr0], %x[mstride1]]\n"
      "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
      "str qU, [%x[outptr0], %x[mstride2]]\n"
      "fsub U.4s, BXTx12.4s, BXTx14.4s\n"
      "str qU, [%x[outptr0], %x[mstride3]]\n"
      "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

      // Second row
      "fneg U.4s, BXTx23.4s\n"
      "str qU, [%x[outptr4]]\n"
      "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
      "str qU, [%x[outptr4], %x[mstride1]]\n"
      "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
      "str qU, [%x[outptr4], %x[mstride2]]\n"
      "fsub U.4s, BXTx22.4s, BXTx24.4s\n"
      "str qU, [%x[outptr4], %x[mstride3]]\n"
      "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

      // Third row
      "fneg U.4s, BXTx33.4s\n"
      "str qU, [%x[outptr8]]\n"
      "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
      "str qU, [%x[outptr8], %x[mstride1]]\n"
      "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
      "str qU, [%x[outptr8], %x[mstride2]]\n"
      "fsub U.4s, BXTx32.4s, BXTx34.4s\n"
      "str qU, [%x[outptr8], %x[mstride3]]\n"
      "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

      // Fourth row, simultaneously load the first column of inputs for the
      // next tile.
      "fneg U.4s, BXTx43.4s\n"
      "str qU, [%x[outptr12]]\n"
      "ldr qx_13, [%x[inptr0]]\n"

      "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
      "str qU, [%x[outptr12], %x[mstride1]]\n"
      "ldr qx_23, [%x[inptr1]]\n"

      "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
      "str qU, [%x[outptr12], %x[mstride2]]\n"
      "ldr qx_33, [%x[inptr2]]\n"

      "fsub U.4s, BXTx42.4s, BXTx44.4s\n"
      "str qU, [%x[outptr12], %x[mstride3]]\n"
      "ldr qx_43, [%x[inptr3]]\n"

      "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"

      // Update the loop counter, subtract two to account for both the head and
      // the tail.
      "subs %x[tile_j], %x[tile_j], #2\n"
      "beq 2f\n"  // Jump to "A" tail if out of tiles

      // ----------------------------------------------------------------------
      "1:"
        // Start part A
        // Load last column of this tile (the first column has already been
        // loaded) and compute latter two columns of X.T x.
        "fsub AXTx13.4s, x_13.4s, x_33.4s\n"
        "ldr qx_14, [%x[inptr0], %x[colstride1]]\n"
        "fadd AXTx23.4s, x_23.4s, x_33.4s\n"
        "ldr qx_24, [%x[inptr1], %x[colstride1]]\n"
        "fsub AXTx33.4s, x_33.4s, x_23.4s\n"
        "ldr qx_34, [%x[inptr2], %x[colstride1]]\n"
        "fsub AXTx43.4s, x_23.4s, x_43.4s\n"
        "ldr qx_44, [%x[inptr3], %x[colstride1]]\n"
        "fsub AXTx14.4s, x_14.4s, x_34.4s\n"
        "add %x[inptr0], %x[inptr0], %x[colstride2]\n"
        "fadd AXTx24.4s, x_24.4s, x_34.4s\n"
        "add %x[inptr1], %x[inptr1], %x[colstride2]\n"
        "fsub AXTx34.4s, x_34.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], %x[colstride2]\n"
        "fsub AXTx44.4s, x_24.4s, x_44.4s\n"
        "add %x[inptr3], %x[inptr3], %x[colstride2]\n"

        // Compute and store U.
        // First row
        "fsub U.4s, AXTx11.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, AXTx12.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, AXTx13.4s, AXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, AXTx12.4s, AXTx14.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

        // Second row
        "fsub U.4s, AXTx21.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, AXTx22.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, AXTx23.4s, AXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fsub U.4s, AXTx22.4s, AXTx24.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

        // Third row
        "fsub U.4s, AXTx31.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, AXTx32.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, AXTx33.4s, AXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, AXTx32.4s, AXTx34.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

        // Fourth row
        "fsub U.4s, AXTx41.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "ldr qx_13, [%x[inptr0]]\n"

        "fadd U.4s, AXTx42.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "ldr qx_23, [%x[inptr1]]\n"

        "fsub U.4s, AXTx43.4s, AXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "ldr qx_33, [%x[inptr2]]\n"

        "fsub U.4s, AXTx42.4s, AXTx44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "ldr qx_43, [%x[inptr3]]\n"

        "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"

        "subs %x[tile_j], %x[tile_j], #1\n"
        "beq 3f\n"  // Jump to 'B' tail

        // Start part B
        // Load last column of this tile (the first column has already been
        // loaded) and compute latter two columns of X.T x.
        "fsub BXTx13.4s, x_13.4s, x_33.4s\n"
        "ldr qx_14, [%x[inptr0], %x[colstride1]]\n"
        "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
        "ldr qx_24, [%x[inptr1], %x[colstride1]]\n"
        "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
        "ldr qx_34, [%x[inptr2], %x[colstride1]]\n"
        "fsub BXTx43.4s, x_23.4s, x_43.4s\n"
        "ldr qx_44, [%x[inptr3], %x[colstride1]]\n"
        "fsub BXTx14.4s, x_14.4s, x_34.4s\n"
        "add %x[inptr0], %x[inptr0], %x[colstride2]\n"
        "fadd BXTx24.4s, x_24.4s, x_34.4s\n"
        "add %x[inptr1], %x[inptr1], %x[colstride2]\n"
        "fsub BXTx34.4s, x_34.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], %x[colstride2]\n"
        "fsub BXTx44.4s, x_24.4s, x_44.4s\n"
        "add %x[inptr3], %x[inptr3], %x[colstride2]\n"

        // Compute and store U.
        // First row
        "fsub U.4s, BXTx11.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, BXTx12.4s, BXTx14.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

        // Second row
        "fsub U.4s, BXTx21.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fsub U.4s, BXTx22.4s, BXTx24.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

        // Third row
        "fsub U.4s, BXTx31.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, BXTx32.4s, BXTx34.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

        // Fourth row
        "fsub U.4s, BXTx41.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "ldr qx_13, [%x[inptr0]]\n"

        "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "ldr qx_23, [%x[inptr1]]\n"

        "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "ldr qx_33, [%x[inptr2]]\n"

        "fsub U.4s, BXTx42.4s, BXTx44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "ldr qx_43, [%x[inptr3]]\n"

        "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"
        "subs %x[tile_j], %x[tile_j], #1\n"
        "bne 1b\n"  // Continue loop, otherwise flow into 'A' tail

      // ----------------------------------------------------------------------
      "2:"
        // 'A' tail
        // Since the final column is padding and the last-but-one column has
        // already been loaded just compute the 3rd column of `X.T x'.
        "fsub AXTx13.4s, x_13.4s, x_33.4s\n"
        "fadd AXTx23.4s, x_23.4s, x_33.4s\n"
        "fsub AXTx33.4s, x_33.4s, x_23.4s\n"
        "fsub AXTx43.4s, x_23.4s, x_43.4s\n"

        // Compute and store U. Modified to account for the final column of X.T
        // x containing padding. Note, it is also unnecessary to update the
        // output pointers.
        // First row
        "fsub U.4s, AXTx11.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, AXTx12.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, AXTx13.4s, AXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "str qAXTx12, [%x[outptr0], %x[mstride3]]\n"

        // Second row
        "fsub U.4s, AXTx21.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, AXTx22.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, AXTx23.4s, AXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "str qAXTx22, [%x[outptr4], %x[mstride3]]\n"

        // Third row
        "fsub U.4s, AXTx31.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, AXTx32.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, AXTx33.4s, AXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "str qAXTx32, [%x[outptr8], %x[mstride3]]\n"

        // Fourth row
        "fsub U.4s, AXTx41.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fadd U.4s, AXTx42.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, AXTx43.4s, AXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "str qAXTx42, [%x[outptr12], %x[mstride3]]\n"

        "b 4f\n"  // Jump to end of function

      // ----------------------------------------------------------------------
      "3:"
        // 'B' tail
        // Since the final column is padding and the last-but-one column has
        // already been loaded just compute the 3rd column of `X.T x'.
        "fsub BXTx13.4s, x_13.4s, x_33.4s\n"
        "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
        "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
        "fsub BXTx43.4s, x_23.4s, x_43.4s\n"

        // Compute and store U. Modified to account for the final column of X.T
        // x containing padding. Note, it is also unnecessary to update the
        // output pointers.
        // First row
        "fsub U.4s, BXTx11.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "str qBXTx12, [%x[outptr0], %x[mstride3]]\n"

        // Second row
        "fsub U.4s, BXTx21.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "str qBXTx22, [%x[outptr4], %x[mstride3]]\n"

        // Third row
        "fsub U.4s, BXTx31.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "str qBXTx32, [%x[outptr8], %x[mstride3]]\n"

        // Fourth row
        "fsub U.4s, BXTx41.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "str qBXTx42, [%x[outptr12], %x[mstride3]]\n"

      // ----------------------------------------------------------------------
      "4:"
        // End of function

      // Clear names
      ".unreq x_13\n" ".unreq qx_13\n" ".unreq x_14\n" ".unreq qx_14\n"
      ".unreq x_23\n" ".unreq qx_23\n" ".unreq x_24\n" ".unreq qx_24\n"
      ".unreq x_33\n" ".unreq qx_33\n" ".unreq x_34\n" ".unreq qx_34\n"
      ".unreq x_43\n" ".unreq qx_43\n" ".unreq x_44\n" ".unreq qx_44\n"
      ".unreq AXTx11\n" ".unreq BXTx13\n"
      ".unreq AXTx12\n" ".unreq BXTx14\n" ".unreq qAXTx12\n"
      ".unreq AXTx21\n" ".unreq BXTx23\n"
      ".unreq AXTx22\n" ".unreq BXTx24\n" ".unreq qAXTx22\n"
      ".unreq AXTx31\n" ".unreq BXTx33\n"
      ".unreq AXTx32\n" ".unreq BXTx34\n" ".unreq qAXTx32\n"
      ".unreq AXTx41\n" ".unreq BXTx43\n"
      ".unreq AXTx42\n" ".unreq BXTx44\n" ".unreq qAXTx42\n"
      ".unreq AXTx13\n" ".unreq BXTx11\n"
      ".unreq AXTx14\n" ".unreq BXTx12\n" ".unreq qBXTx12\n"
      ".unreq AXTx23\n" ".unreq BXTx21\n"
      ".unreq AXTx24\n" ".unreq BXTx22\n" ".unreq qBXTx22\n"
      ".unreq AXTx33\n" ".unreq BXTx31\n"
      ".unreq AXTx34\n" ".unreq BXTx32\n" ".unreq qBXTx32\n"
      ".unreq AXTx43\n" ".unreq BXTx41\n"
      ".unreq AXTx44\n" ".unreq BXTx42\n" ".unreq qBXTx42\n"
      ".unreq U\n" ".unreq qU\n"
    : [inptr0] "+r" (inptr0),
      [inptr1] "+r" (inptr1),
      [inptr2] "+r" (inptr2),
      [inptr3] "+r" (inptr3),
      [outptr0] "+r" (outptr0),
      [outptr4] "+r" (outptr4),
      [outptr8] "+r" (outptr8),
      [outptr12] "+r" (outptr12),
      [tile_j] "+r" (tile_j)  // Tile counter
    : [colstride1] "r" (1 * input_col_stride * sizeof(float)),
      [colstride2] "r" (2 * input_col_stride * sizeof(float)),
      [colstride3] "r" (3 * input_col_stride * sizeof(float)),
      [mstride1] "r" (1 * matrix_stride * sizeof(float)),
      [mstride2] "r" (2 * matrix_stride * sizeof(float)),
      [mstride3] "r" (3 * matrix_stride * sizeof(float)),
      [matrix_row_stride] "r" (matrix_row_stride * sizeof(float))
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24"
  );
}

// Pad top, left and right by 1.
template <>
template <>
inline void Winograd2x2_3x3GemmInput<float>::process_tile_row<1, 1, 0, 1, 4>(
    const int tile_N,
    const float* const input,
    const int input_row_stride,
    const int input_col_stride,
    float* const matrix,
    const int matrix_stride,
    const int matrix_row_stride
) {
  const float *inptr0 = input;
  const float *inptr1 = input + input_row_stride;
  const float *inptr2 = input + input_row_stride * 2;

  float *outptr0 = matrix;
  float *outptr4 = matrix + matrix_stride * 4;
  float *outptr8 = matrix + matrix_stride * 8;
  float *outptr12 = matrix + matrix_stride * 12;

  int tile_j = tile_N;  // Tiles to process

  asm volatile (
      // Named SIMD registers according to the policy given above
      // Registers into which to load the latter two columns of `x`
      // NOTE: We need only load the latter three rows since we know that the
      // first row is padded.
      "x_23 .req v1\n qx_23 .req q1\n" "x_24 .req v5\n qx_24 .req q5\n"
      "x_33 .req v2\n qx_33 .req q2\n" "x_34 .req v6\n qx_34 .req q6\n"
      "x_43 .req v3\n qx_43 .req q3\n" "x_44 .req v7\n qx_44 .req q7\n"

      // Registers for storing X.T x (both A and B halves)
      "AXTx11 .req  v8\n" "BXTx13 .req  v8\n"
      "AXTx12 .req  v9\n" "BXTx14 .req  v9\n" "qAXTx12 .req  q9\n"
      "AXTx21 .req v10\n" "BXTx23 .req v10\n"
      "AXTx22 .req v11\n" "BXTx24 .req v11\n" "qAXTx22 .req q11\n"
      "AXTx31 .req v12\n" "BXTx33 .req v12\n"
      "AXTx32 .req v13\n" "BXTx34 .req v13\n" "qAXTx32 .req q13\n"
      "AXTx41 .req v14\n" "BXTx43 .req v14\n"
      "AXTx42 .req v15\n" "BXTx44 .req v15\n" "qAXTx42 .req q15\n"
      "AXTx13 .req v16\n" "BXTx11 .req v16\n"
      "AXTx14 .req v17\n" "BXTx12 .req v17\n" "qBXTx12 .req q17\n"
      "AXTx23 .req v18\n" "BXTx21 .req v18\n"
      "AXTx24 .req v19\n" "BXTx22 .req v19\n" "qBXTx22 .req q19\n"
      "AXTx33 .req v20\n" "BXTx31 .req v20\n"
      "AXTx34 .req v21\n" "BXTx32 .req v21\n" "qBXTx32 .req q21\n"
      "AXTx43 .req v22\n" "BXTx41 .req v22\n"
      "AXTx44 .req v23\n" "BXTx42 .req v23\n" "qBXTx42 .req q23\n"

      // Result register (TODO Does using more registers yield better
      // performance)
      "U .req v24\n qU .req q24\n"

      // ----------------------------------------------------------------------
      // Head of loop
      //   Loads a complete 4x4 tile of x, computes X.T x, computes and stores
      //   `U = X.T x X`. Prepares for the 'A' half of the loop.
      //   NOTE: Since the first tile has the leftmost column padded we can
      //   skip 4 loads and 4 calculations for the matrix X.T x X.

      // Temporarily alias registers for computing the first (non-padded)
      // column of x.
      "x_22 .req v1\n qx_22 .req q1\n"
      "x_32 .req v2\n qx_32 .req q2\n"
      "x_42 .req v3\n qx_42 .req q3\n"

      "ldr qx_22, [%x[inptr1]]\n"
      "ldr qx_32, [%x[inptr2]]\n"
      "ldr qx_42, [%x[inptr3]]\n"

      "fneg BXTx12.4s,          x_32.4s\n"
      "fadd BXTx22.4s, x_22.4s, x_32.4s\n"
      "fsub BXTx32.4s, x_32.4s, x_22.4s\n"
      "fsub BXTx42.4s, x_22.4s, x_42.4s\n"

      ".unreq x_22\n .unreq qx_22\n"
      ".unreq x_32\n .unreq qx_32\n"
      ".unreq x_42\n .unreq qx_42\n"

      // Load and compute latter two columns of the first tile. Progress the
      // input pointers (by three columns so that the each points are the
      // second column of the next tile, that is, each points at the first
      // column which must be read for the next tile.
      "ldr qx_23, [%x[inptr1], %x[colstride1]]\n"
      "ldr qx_33, [%x[inptr2], %x[colstride1]]\n"
      "ldr qx_43, [%x[inptr3], %x[colstride1]]\n"

      "fneg BXTx13.4s,          x_33.4s\n"

      "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
      "ldr qx_24, [%x[inptr1], %x[colstride2]]\n"

      "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
      "ldr qx_34, [%x[inptr2], %x[colstride2]]\n"

      "fsub BXTx43.4s, x_23.4s, x_43.4s\n"
      "ldr qx_44, [%x[inptr3], %x[colstride2]]\n"

      "fneg BXTx14.4s,          x_34.4s\n"

      "fadd BXTx24.4s, x_24.4s, x_34.4s\n"
      "add %x[inptr1], %x[inptr1], %x[colstride3]\n"

      "fsub BXTx34.4s, x_34.4s, x_24.4s\n"
      "add %x[inptr2], %x[inptr2], %x[colstride3]\n"

      "fsub BXTx44.4s, x_24.4s, x_44.4s\n"
      "add %x[inptr3], %x[inptr3], %x[colstride3]\n"

      // Compute and store U for the first tile
      // First row
      "fneg U.4s, BXTx13.4s\n"
      "str qU, [%x[outptr0]]\n"
      "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
      "str qU, [%x[outptr0], %x[mstride1]]\n"
      "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
      "str qU, [%x[outptr0], %x[mstride2]]\n"
      "fsub U.4s, BXTx12.4s, BXTx14.4s\n"
      "str qU, [%x[outptr0], %x[mstride3]]\n"
      "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

      // Second row
      "fneg U.4s, BXTx23.4s\n"
      "str qU, [%x[outptr4]]\n"
      "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
      "str qU, [%x[outptr4], %x[mstride1]]\n"
      "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
      "str qU, [%x[outptr4], %x[mstride2]]\n"
      "fsub U.4s, BXTx22.4s, BXTx24.4s\n"
      "str qU, [%x[outptr4], %x[mstride3]]\n"
      "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

      // Third row
      "fneg U.4s, BXTx33.4s\n"
      "str qU, [%x[outptr8]]\n"
      "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
      "str qU, [%x[outptr8], %x[mstride1]]\n"
      "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
      "str qU, [%x[outptr8], %x[mstride2]]\n"
      "fsub U.4s, BXTx32.4s, BXTx34.4s\n"
      "str qU, [%x[outptr8], %x[mstride3]]\n"
      "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

      // Fourth row, simultaneously load the first column of inputs for the
      // next tile.
      "fneg U.4s, BXTx43.4s\n"
      "str qU, [%x[outptr12]]\n"

      "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
      "str qU, [%x[outptr12], %x[mstride1]]\n"
      "ldr qx_23, [%x[inptr1]]\n"

      "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
      "str qU, [%x[outptr12], %x[mstride2]]\n"
      "ldr qx_33, [%x[inptr2]]\n"

      "fsub U.4s, BXTx42.4s, BXTx44.4s\n"
      "str qU, [%x[outptr12], %x[mstride3]]\n"
      "ldr qx_43, [%x[inptr3]]\n"

      "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"

      // Update the loop counter, subtract two to account for both the head and
      // the tail.
      "subs %x[tile_j], %x[tile_j], #2\n"
      "beq 2f\n"  // Jump to "A" tail if out of tiles

      // ----------------------------------------------------------------------
      "1:"
        // Start part A
        // Load last column of this tile (the first column has already been
        // loaded) and compute latter two columns of X.T x.
        "fneg AXTx13.4s,          x_33.4s\n"
        "fadd AXTx23.4s, x_23.4s, x_33.4s\n"
        "ldr qx_24, [%x[inptr1], %x[colstride1]]\n"
        "fsub AXTx33.4s, x_33.4s, x_23.4s\n"
        "ldr qx_34, [%x[inptr2], %x[colstride1]]\n"
        "fsub AXTx43.4s, x_23.4s, x_43.4s\n"
        "ldr qx_44, [%x[inptr3], %x[colstride1]]\n"
        "fneg AXTx14.4s,          x_34.4s\n"
        "fadd AXTx24.4s, x_24.4s, x_34.4s\n"
        "add %x[inptr1], %x[inptr1], %x[colstride2]\n"
        "fsub AXTx34.4s, x_34.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], %x[colstride2]\n"
        "fsub AXTx44.4s, x_24.4s, x_44.4s\n"
        "add %x[inptr3], %x[inptr3], %x[colstride2]\n"

        // Compute and store U.
        // First row
        "fsub U.4s, AXTx11.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, AXTx12.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, AXTx13.4s, AXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, AXTx12.4s, AXTx14.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

        // Second row
        "fsub U.4s, AXTx21.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, AXTx22.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, AXTx23.4s, AXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fsub U.4s, AXTx22.4s, AXTx24.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

        // Third row
        "fsub U.4s, AXTx31.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, AXTx32.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, AXTx33.4s, AXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, AXTx32.4s, AXTx34.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

        // Fourth row
        "fsub U.4s, AXTx41.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"

        "fadd U.4s, AXTx42.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "ldr qx_23, [%x[inptr1]]\n"

        "fsub U.4s, AXTx43.4s, AXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "ldr qx_33, [%x[inptr2]]\n"

        "fsub U.4s, AXTx42.4s, AXTx44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "ldr qx_43, [%x[inptr3]]\n"

        "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"

        "subs %x[tile_j], %x[tile_j], #1\n"
        "beq 3f\n"  // Jump to 'B' tail

        // Start part B
        // Load last column of this tile (the first column has already been
        // loaded) and compute latter two columns of X.T x.
        "fneg BXTx13.4s,          x_33.4s\n"
        "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
        "ldr qx_24, [%x[inptr1], %x[colstride1]]\n"
        "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
        "ldr qx_34, [%x[inptr2], %x[colstride1]]\n"
        "fsub BXTx43.4s, x_23.4s, x_43.4s\n"
        "ldr qx_44, [%x[inptr3], %x[colstride1]]\n"
        "fneg BXTx14.4s,          x_34.4s\n"
        "fadd BXTx24.4s, x_24.4s, x_34.4s\n"
        "add %x[inptr1], %x[inptr1], %x[colstride2]\n"
        "fsub BXTx34.4s, x_34.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], %x[colstride2]\n"
        "fsub BXTx44.4s, x_24.4s, x_44.4s\n"
        "add %x[inptr3], %x[inptr3], %x[colstride2]\n"

        // Compute and store U.
        // First row
        "fsub U.4s, BXTx11.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, BXTx12.4s, BXTx14.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

        // Second row
        "fsub U.4s, BXTx21.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fsub U.4s, BXTx22.4s, BXTx24.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

        // Third row
        "fsub U.4s, BXTx31.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, BXTx32.4s, BXTx34.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

        // Fourth row
        "fsub U.4s, BXTx41.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"

        "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "ldr qx_23, [%x[inptr1]]\n"

        "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "ldr qx_33, [%x[inptr2]]\n"

        "fsub U.4s, BXTx42.4s, BXTx44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"
        "ldr qx_43, [%x[inptr3]]\n"

        "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"
        "subs %x[tile_j], %x[tile_j], #1\n"
        "bne 1b\n"  // Continue loop, otherwise flow into 'A' tail

      // ----------------------------------------------------------------------
      "2:"
        // 'A' tail
        // Since the final column is padding and the last-but-one column has
        // already been loaded just compute the 3rd column of `X.T x'.
        "fneg AXTx13.4s,          x_33.4s\n"
        "fadd AXTx23.4s, x_23.4s, x_33.4s\n"
        "fsub AXTx33.4s, x_33.4s, x_23.4s\n"
        "fsub AXTx43.4s, x_23.4s, x_43.4s\n"

        // Compute and store U. Modified to account for the final column of X.T
        // x containing padding. Note, it is also unnecessary to update the
        // output pointers.
        // First row
        "fsub U.4s, AXTx11.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, AXTx12.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, AXTx13.4s, AXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "str qAXTx12, [%x[outptr0], %x[mstride3]]\n"

        // Second row
        "fsub U.4s, AXTx21.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, AXTx22.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, AXTx23.4s, AXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "str qAXTx22, [%x[outptr4], %x[mstride3]]\n"

        // Third row
        "fsub U.4s, AXTx31.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, AXTx32.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, AXTx33.4s, AXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "str qAXTx32, [%x[outptr8], %x[mstride3]]\n"

        // Fourth row
        "fsub U.4s, AXTx41.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fadd U.4s, AXTx42.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, AXTx43.4s, AXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "str qAXTx42, [%x[outptr12], %x[mstride3]]\n"

        "b 4f\n"  // Jump to end of function

      // ----------------------------------------------------------------------
      "3:"
        // 'B' tail
        // Since the final column is padding and the last-but-one column has
        // already been loaded just compute the 3rd column of `X.T x'.
        "fneg BXTx13.4s,          x_33.4s\n"
        "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
        "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
        "fsub BXTx43.4s, x_23.4s, x_43.4s\n"

        // Compute and store U. Modified to account for the final column of X.T
        // x containing padding. Note, it is also unnecessary to update the
        // output pointers.
        // First row
        "fsub U.4s, BXTx11.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "str qBXTx12, [%x[outptr0], %x[mstride3]]\n"

        // Second row
        "fsub U.4s, BXTx21.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "str qBXTx22, [%x[outptr4], %x[mstride3]]\n"

        // Third row
        "fsub U.4s, BXTx31.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "str qBXTx32, [%x[outptr8], %x[mstride3]]\n"

        // Fourth row
        "fsub U.4s, BXTx41.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "str qBXTx42, [%x[outptr12], %x[mstride3]]\n"

      // ----------------------------------------------------------------------
      "4:"
        // End of function

      // Clear names
      ".unreq x_23\n" ".unreq qx_23\n" ".unreq x_24\n" ".unreq qx_24\n"
      ".unreq x_33\n" ".unreq qx_33\n" ".unreq x_34\n" ".unreq qx_34\n"
      ".unreq x_43\n" ".unreq qx_43\n" ".unreq x_44\n" ".unreq qx_44\n"
      ".unreq AXTx11\n" ".unreq BXTx13\n"
      ".unreq AXTx12\n" ".unreq BXTx14\n" ".unreq qAXTx12\n"
      ".unreq AXTx21\n" ".unreq BXTx23\n"
      ".unreq AXTx22\n" ".unreq BXTx24\n" ".unreq qAXTx22\n"
      ".unreq AXTx31\n" ".unreq BXTx33\n"
      ".unreq AXTx32\n" ".unreq BXTx34\n" ".unreq qAXTx32\n"
      ".unreq AXTx41\n" ".unreq BXTx43\n"
      ".unreq AXTx42\n" ".unreq BXTx44\n" ".unreq qAXTx42\n"
      ".unreq AXTx13\n" ".unreq BXTx11\n"
      ".unreq AXTx14\n" ".unreq BXTx12\n" ".unreq qBXTx12\n"
      ".unreq AXTx23\n" ".unreq BXTx21\n"
      ".unreq AXTx24\n" ".unreq BXTx22\n" ".unreq qBXTx22\n"
      ".unreq AXTx33\n" ".unreq BXTx31\n"
      ".unreq AXTx34\n" ".unreq BXTx32\n" ".unreq qBXTx32\n"
      ".unreq AXTx43\n" ".unreq BXTx41\n"
      ".unreq AXTx44\n" ".unreq BXTx42\n" ".unreq qBXTx42\n"
      ".unreq U\n" ".unreq qU\n"
    : [inptr1] "+r" (inptr0),  // Offset to account for padded row
      [inptr2] "+r" (inptr1),  // Offset to account for padded row
      [inptr3] "+r" (inptr2),  // Offset to account for padded row
      [outptr0] "+r" (outptr0),
      [outptr4] "+r" (outptr4),
      [outptr8] "+r" (outptr8),
      [outptr12] "+r" (outptr12),
      [tile_j] "+r" (tile_j)  // Tile counter
    : [colstride1] "r" (1 * input_col_stride * sizeof(float)),
      [colstride2] "r" (2 * input_col_stride * sizeof(float)),
      [colstride3] "r" (3 * input_col_stride * sizeof(float)),
      [mstride1] "r" (1 * matrix_stride * sizeof(float)),
      [mstride2] "r" (2 * matrix_stride * sizeof(float)),
      [mstride3] "r" (3 * matrix_stride * sizeof(float)),
      [matrix_row_stride] "r" (matrix_row_stride * sizeof(float))
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24"
  );
}

// Pad left, right and bottom by 1.
template <>
template <>
inline void Winograd2x2_3x3GemmInput<float>::process_tile_row<0, 1, 1, 1, 4>(
    const int tile_N,
    const float* const input,
    const int input_row_stride,
    const int input_col_stride,
    float* const matrix,
    const int matrix_stride,
    const int matrix_row_stride
) {
  const float *inptr0 = input;
  const float *inptr1 = input + input_row_stride;
  const float *inptr2 = input + input_row_stride * 2;

  float *outptr0 = matrix;
  float *outptr4 = matrix + matrix_stride * 4;
  float *outptr8 = matrix + matrix_stride * 8;
  float *outptr12 = matrix + matrix_stride * 12;

  int tile_j = tile_N;  // Tiles to process

  asm volatile (
      // Named SIMD registers according to the policy given above
      // Registers into which to load the latter two columns of `x`
      // NOTE: Bottom row is not required since since it is padded.
      "x_13 .req v0\n qx_13 .req q0\n" "x_14 .req v4\n qx_14 .req q4\n"
      "x_23 .req v1\n qx_23 .req q1\n" "x_24 .req v5\n qx_24 .req q5\n"
      "x_33 .req v2\n qx_33 .req q2\n" "x_34 .req v6\n qx_34 .req q6\n"

      // Registers for storing X.T x (both A and B halves)
      "AXTx11 .req  v8\n" "BXTx13 .req  v8\n"
      "AXTx12 .req  v9\n" "BXTx14 .req  v9\n" "qAXTx12 .req  q9\n"
      "AXTx21 .req v10\n" "BXTx23 .req v10\n"
      "AXTx22 .req v11\n" "BXTx24 .req v11\n" "qAXTx22 .req q11\n"
      "AXTx31 .req v12\n" "BXTx33 .req v12\n"
      "AXTx32 .req v13\n" "BXTx34 .req v13\n" "qAXTx32 .req q13\n"
      "AXTx41 .req v14\n" "BXTx43 .req v14\n"
      "AXTx42 .req v15\n" "BXTx44 .req v15\n" "qAXTx42 .req q15\n"
      "AXTx13 .req v16\n" "BXTx11 .req v16\n"
      "AXTx14 .req v17\n" "BXTx12 .req v17\n" "qBXTx12 .req q17\n"
      "AXTx23 .req v18\n" "BXTx21 .req v18\n"
      "AXTx24 .req v19\n" "BXTx22 .req v19\n" "qBXTx22 .req q19\n"
      "AXTx33 .req v20\n" "BXTx31 .req v20\n"
      "AXTx34 .req v21\n" "BXTx32 .req v21\n" "qBXTx32 .req q21\n"
      "AXTx43 .req v22\n" "BXTx41 .req v22\n"
      "AXTx44 .req v23\n" "BXTx42 .req v23\n" "qBXTx42 .req q23\n"

      // Result register (TODO Does using more registers yield better
      // performance)
      "U .req v24\n qU .req q24\n"

      // ----------------------------------------------------------------------
      // Head of loop
      //   Loads a complete 4x4 tile of x, computes X.T x, computes and stores
      //   `U = X.T x X`. Prepares for the 'A' half of the loop.
      //   NOTE: Since the first tile has the leftmost column padded we can
      //   skip 4 loads and 4 calculations for the matrix X.T x X.

      // Temporarily alias registers for computing the first (non-padded)
      // column of x.
      "x_12 .req v0\n qx_12 .req q0\n"
      "x_22 .req v1\n qx_22 .req q1\n"
      "x_32 .req v2\n qx_32 .req q2\n"

      "ldr qx_12, [%x[inptr0]]\n"
      "ldr qx_22, [%x[inptr1]]\n"
      "ldr qx_32, [%x[inptr2]]\n"

      "fsub BXTx12.4s,  x_12.4s, x_32.4s\n"
      "fadd BXTx22.4s,  x_22.4s, x_32.4s\n"
      "fsub BXTx32.4s,  x_32.4s, x_22.4s\n"
      "mov  BXTx42.16b, x_22.16b\n"  // Probably should do better

      ".unreq x_12\n .unreq qx_12\n"
      ".unreq x_22\n .unreq qx_22\n"
      ".unreq x_32\n .unreq qx_32\n"

      // Load and compute latter two columns of the first tile. Progress the
      // input pointers (by three columns so that the each points are the
      // second column of the next tile, that is, each points at the first
      // column which must be read for the next tile.
      "ldr qx_13, [%x[inptr0], %x[colstride1]]\n"
      "ldr qx_23, [%x[inptr1], %x[colstride1]]\n"
      "ldr qx_33, [%x[inptr2], %x[colstride1]]\n"

      "fsub BXTx13.4s, x_13.4s, x_33.4s\n"
      "ldr qx_14, [%x[inptr0], %x[colstride2]]\n"

      "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
      "ldr qx_24, [%x[inptr1], %x[colstride2]]\n"

      "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
      "ldr qx_34, [%x[inptr2], %x[colstride2]]\n"

      "mov  BXTx43.16b, x_23.16b\n"
      "fsub BXTx14.4s,  x_14.4s, x_34.4s\n"
      "add %x[inptr0],  %x[inptr0], %x[colstride3]\n"

      "fadd BXTx24.4s, x_24.4s, x_34.4s\n"
      "add %x[inptr1], %x[inptr1], %x[colstride3]\n"

      "fsub BXTx34.4s, x_34.4s, x_24.4s\n"
      "add %x[inptr2], %x[inptr2], %x[colstride3]\n"

      "mov BXTx44.16b, x_24.16b\n"

      // Compute and store U for the first tile
      // First row
      "fneg U.4s, BXTx13.4s\n"
      "str qU, [%x[outptr0]]\n"
      "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
      "str qU, [%x[outptr0], %x[mstride1]]\n"
      "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
      "str qU, [%x[outptr0], %x[mstride2]]\n"
      "fsub U.4s, BXTx12.4s, BXTx14.4s\n"
      "str qU, [%x[outptr0], %x[mstride3]]\n"
      "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

      // Second row
      "fneg U.4s, BXTx23.4s\n"
      "str qU, [%x[outptr4]]\n"
      "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
      "str qU, [%x[outptr4], %x[mstride1]]\n"
      "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
      "str qU, [%x[outptr4], %x[mstride2]]\n"
      "fsub U.4s, BXTx22.4s, BXTx24.4s\n"
      "str qU, [%x[outptr4], %x[mstride3]]\n"
      "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

      // Third row
      "fneg U.4s, BXTx33.4s\n"
      "str qU, [%x[outptr8]]\n"
      "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
      "str qU, [%x[outptr8], %x[mstride1]]\n"
      "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
      "str qU, [%x[outptr8], %x[mstride2]]\n"
      "fsub U.4s, BXTx32.4s, BXTx34.4s\n"
      "str qU, [%x[outptr8], %x[mstride3]]\n"
      "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

      // Fourth row, simultaneously load the first column of inputs for the
      // next tile.
      "fneg U.4s, BXTx43.4s\n"
      "str qU, [%x[outptr12]]\n"
      "ldr qx_13, [%x[inptr0]]\n"

      "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
      "str qU, [%x[outptr12], %x[mstride1]]\n"
      "ldr qx_23, [%x[inptr1]]\n"

      "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
      "str qU, [%x[outptr12], %x[mstride2]]\n"
      "ldr qx_33, [%x[inptr2]]\n"

      "fsub U.4s, BXTx42.4s, BXTx44.4s\n"
      "str qU, [%x[outptr12], %x[mstride3]]\n"

      "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"

      // Update the loop counter, subtract two to account for both the head and
      // the tail.
      "subs %x[tile_j], %x[tile_j], #2\n"
      "beq 2f\n"  // Jump to "A" tail if out of tiles

      // ----------------------------------------------------------------------
      "1:"
        // Start part A
        // Load last column of this tile (the first column has already been
        // loaded) and compute latter two columns of X.T x.
        "fsub AXTx13.4s, x_13.4s, x_33.4s\n"
        "ldr qx_14, [%x[inptr0], %x[colstride1]]\n"
        "fadd AXTx23.4s, x_23.4s, x_33.4s\n"
        "ldr qx_24, [%x[inptr1], %x[colstride1]]\n"
        "fsub AXTx33.4s, x_33.4s, x_23.4s\n"
        "ldr qx_34, [%x[inptr2], %x[colstride1]]\n"
        "mov  AXTx43.16b, x_23.16b\n"

        "fsub AXTx14.4s, x_14.4s, x_34.4s\n"
        "add %x[inptr0], %x[inptr0], %x[colstride2]\n"
        "fadd AXTx24.4s, x_24.4s, x_34.4s\n"
        "add %x[inptr1], %x[inptr1], %x[colstride2]\n"
        "fsub AXTx34.4s, x_34.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], %x[colstride2]\n"
        "mov  AXTx44.16b, x_24.16b\n"

        // Compute and store U.
        // First row
        "fsub U.4s, AXTx11.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, AXTx12.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, AXTx13.4s, AXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, AXTx12.4s, AXTx14.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

        // Second row
        "fsub U.4s, AXTx21.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, AXTx22.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, AXTx23.4s, AXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fsub U.4s, AXTx22.4s, AXTx24.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

        // Third row
        "fsub U.4s, AXTx31.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, AXTx32.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, AXTx33.4s, AXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, AXTx32.4s, AXTx34.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

        // Fourth row
        "fsub U.4s, AXTx41.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "ldr qx_13, [%x[inptr0]]\n"

        "fadd U.4s, AXTx42.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "ldr qx_23, [%x[inptr1]]\n"

        "fsub U.4s, AXTx43.4s, AXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "ldr qx_33, [%x[inptr2]]\n"

        "fsub U.4s, AXTx42.4s, AXTx44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"

        "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"

        "subs %x[tile_j], %x[tile_j], #1\n"
        "beq 3f\n"  // Jump to 'B' tail

        // Start part B
        // Load last column of this tile (the first column has already been
        // loaded) and compute latter two columns of X.T x.
        "fsub BXTx13.4s, x_13.4s, x_33.4s\n"
        "ldr qx_14, [%x[inptr0], %x[colstride1]]\n"
        "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
        "ldr qx_24, [%x[inptr1], %x[colstride1]]\n"
        "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
        "ldr qx_34, [%x[inptr2], %x[colstride1]]\n"
        "mov BXTx43.16b, x_23.16b\n"

        "fsub BXTx14.4s, x_14.4s, x_34.4s\n"
        "add %x[inptr0], %x[inptr0], %x[colstride2]\n"
        "fadd BXTx24.4s, x_24.4s, x_34.4s\n"
        "add %x[inptr1], %x[inptr1], %x[colstride2]\n"
        "fsub BXTx34.4s, x_34.4s, x_24.4s\n"
        "add %x[inptr2], %x[inptr2], %x[colstride2]\n"
        "mov BXTx44.16b, x_24.16b\n"

        // Compute and store U.
        // First row
        "fsub U.4s, BXTx11.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "fsub U.4s, BXTx12.4s, BXTx14.4s\n"
        "str qU, [%x[outptr0], %x[mstride3]]\n"
        "add %x[outptr0], %x[outptr0], %x[matrix_row_stride]\n"

        // Second row
        "fsub U.4s, BXTx21.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "fsub U.4s, BXTx22.4s, BXTx24.4s\n"
        "str qU, [%x[outptr4], %x[mstride3]]\n"
        "add %x[outptr4], %x[outptr4], %x[matrix_row_stride]\n"

        // Third row
        "fsub U.4s, BXTx31.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "fsub U.4s, BXTx32.4s, BXTx34.4s\n"
        "str qU, [%x[outptr8], %x[mstride3]]\n"
        "add %x[outptr8], %x[outptr8], %x[matrix_row_stride]\n"

        // Fourth row
        "fsub U.4s, BXTx41.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "ldr qx_13, [%x[inptr0]]\n"

        "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "ldr qx_23, [%x[inptr1]]\n"

        "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "ldr qx_33, [%x[inptr2]]\n"

        "fsub U.4s, BXTx42.4s, BXTx44.4s\n"
        "str qU, [%x[outptr12], %x[mstride3]]\n"

        "add %x[outptr12], %x[outptr12], %x[matrix_row_stride]\n"
        "subs %x[tile_j], %x[tile_j], #1\n"
        "bne 1b\n"  // Continue loop, otherwise flow into 'A' tail

      // ----------------------------------------------------------------------
      "2:"
        // 'A' tail
        // Since the final column is padding and the last-but-one column has
        // already been loaded just compute the 3rd column of `X.T x'.
        "fsub AXTx13.4s, x_13.4s, x_33.4s\n"
        "fadd AXTx23.4s, x_23.4s, x_33.4s\n"
        "fsub AXTx33.4s, x_33.4s, x_23.4s\n"
        "mov  AXTx43.16b, x_23.16b\n"

        // Compute and store U. Modified to account for the final column of X.T
        // x containing padding. Note, it is also unnecessary to update the
        // output pointers.
        // First row
        "fsub U.4s, AXTx11.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, AXTx12.4s, AXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, AXTx13.4s, AXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "str qAXTx12, [%x[outptr0], %x[mstride3]]\n"

        // Second row
        "fsub U.4s, AXTx21.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, AXTx22.4s, AXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, AXTx23.4s, AXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "str qAXTx22, [%x[outptr4], %x[mstride3]]\n"

        // Third row
        "fsub U.4s, AXTx31.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, AXTx32.4s, AXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, AXTx33.4s, AXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "str qAXTx32, [%x[outptr8], %x[mstride3]]\n"

        // Fourth row
        "fsub U.4s, AXTx41.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fadd U.4s, AXTx42.4s, AXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, AXTx43.4s, AXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "str qAXTx42, [%x[outptr12], %x[mstride3]]\n"

        "b 4f\n"  // Jump to end of function

      // ----------------------------------------------------------------------
      "3:"
        // 'B' tail
        // Since the final column is padding and the last-but-one column has
        // already been loaded just compute the 3rd column of `X.T x'.
        "fsub BXTx13.4s, x_13.4s, x_33.4s\n"
        "fadd BXTx23.4s, x_23.4s, x_33.4s\n"
        "fsub BXTx33.4s, x_33.4s, x_23.4s\n"
        "mov  BXTx43.16b, x_23.16b\n"

        // Compute and store U. Modified to account for the final column of X.T
        // x containing padding. Note, it is also unnecessary to update the
        // output pointers.
        // First row
        "fsub U.4s, BXTx11.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0]]\n"
        "fadd U.4s, BXTx12.4s, BXTx13.4s\n"
        "str qU, [%x[outptr0], %x[mstride1]]\n"
        "fsub U.4s, BXTx13.4s, BXTx12.4s\n"
        "str qU, [%x[outptr0], %x[mstride2]]\n"
        "str qBXTx12, [%x[outptr0], %x[mstride3]]\n"

        // Second row
        "fsub U.4s, BXTx21.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4]]\n"
        "fadd U.4s, BXTx22.4s, BXTx23.4s\n"
        "str qU, [%x[outptr4], %x[mstride1]]\n"
        "fsub U.4s, BXTx23.4s, BXTx22.4s\n"
        "str qU, [%x[outptr4], %x[mstride2]]\n"
        "str qBXTx22, [%x[outptr4], %x[mstride3]]\n"

        // Third row
        "fsub U.4s, BXTx31.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8]]\n"
        "fadd U.4s, BXTx32.4s, BXTx33.4s\n"
        "str qU, [%x[outptr8], %x[mstride1]]\n"
        "fsub U.4s, BXTx33.4s, BXTx32.4s\n"
        "str qU, [%x[outptr8], %x[mstride2]]\n"
        "str qBXTx32, [%x[outptr8], %x[mstride3]]\n"

        // Fourth row
        "fsub U.4s, BXTx41.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12]]\n"
        "fadd U.4s, BXTx42.4s, BXTx43.4s\n"
        "str qU, [%x[outptr12], %x[mstride1]]\n"
        "fsub U.4s, BXTx43.4s, BXTx42.4s\n"
        "str qU, [%x[outptr12], %x[mstride2]]\n"
        "str qBXTx42, [%x[outptr12], %x[mstride3]]\n"

      // ----------------------------------------------------------------------
      "4:"
        // End of function

      // Clear names
      ".unreq x_13\n" ".unreq qx_13\n" ".unreq x_14\n" ".unreq qx_14\n"
      ".unreq x_23\n" ".unreq qx_23\n" ".unreq x_24\n" ".unreq qx_24\n"
      ".unreq x_33\n" ".unreq qx_33\n" ".unreq x_34\n" ".unreq qx_34\n"
      ".unreq AXTx11\n" ".unreq BXTx13\n"
      ".unreq AXTx12\n" ".unreq BXTx14\n" ".unreq qAXTx12\n"
      ".unreq AXTx21\n" ".unreq BXTx23\n"
      ".unreq AXTx22\n" ".unreq BXTx24\n" ".unreq qAXTx22\n"
      ".unreq AXTx31\n" ".unreq BXTx33\n"
      ".unreq AXTx32\n" ".unreq BXTx34\n" ".unreq qAXTx32\n"
      ".unreq AXTx41\n" ".unreq BXTx43\n"
      ".unreq AXTx42\n" ".unreq BXTx44\n" ".unreq qAXTx42\n"
      ".unreq AXTx13\n" ".unreq BXTx11\n"
      ".unreq AXTx14\n" ".unreq BXTx12\n" ".unreq qBXTx12\n"
      ".unreq AXTx23\n" ".unreq BXTx21\n"
      ".unreq AXTx24\n" ".unreq BXTx22\n" ".unreq qBXTx22\n"
      ".unreq AXTx33\n" ".unreq BXTx31\n"
      ".unreq AXTx34\n" ".unreq BXTx32\n" ".unreq qBXTx32\n"
      ".unreq AXTx43\n" ".unreq BXTx41\n"
      ".unreq AXTx44\n" ".unreq BXTx42\n" ".unreq qBXTx42\n"
      ".unreq U\n" ".unreq qU\n"
    : [inptr0] "+r" (inptr0),
      [inptr1] "+r" (inptr1),
      [inptr2] "+r" (inptr2),
      [outptr0] "+r" (outptr0),
      [outptr4] "+r" (outptr4),
      [outptr8] "+r" (outptr8),
      [outptr12] "+r" (outptr12),
      [tile_j] "+r" (tile_j)  // Tile counter
    : [colstride1] "r" (1 * input_col_stride * sizeof(float)),
      [colstride2] "r" (2 * input_col_stride * sizeof(float)),
      [colstride3] "r" (3 * input_col_stride * sizeof(float)),
      [mstride1] "r" (1 * matrix_stride * sizeof(float)),
      [mstride2] "r" (2 * matrix_stride * sizeof(float)),
      [mstride3] "r" (3 * matrix_stride * sizeof(float)),
      [matrix_row_stride] "r" (matrix_row_stride * sizeof(float))
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24"
  );
}
}
#endif  // __aarch64__
