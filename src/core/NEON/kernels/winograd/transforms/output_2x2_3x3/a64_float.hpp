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

/* Float implementation for AArch64.
 */
#ifdef __aarch64__
namespace winograd {


template <>
template <>
inline void Winograd2x2_3x3GemmOutput<float>::_execute<false, false, 0>(
    const Tensor4DShape &output_shape,
    float *output,
    const float *input,
    const int mstride,
    const int matrix_row_stride
) {
  const int tile_M = output_shape.n_rows / 2;
  const int tile_N = output_shape.n_cols / 2;
  int batch = output_shape.n_batches;
  float *outptr = output;

  const float *inptr0 = input;
  const float *inptr4 = input + 4 * mstride;
  const float *inptr8 = input + 8 * mstride;
  const float *inptr12 = input + 12 * mstride;

  const size_t col_stride = sizeof(float) * output_shape.n_channels;
  const size_t row_stride = col_stride * tile_N * 2;

  asm volatile (
      // Aliases for elements of the input matrix `F`
      // V-register      Q-register
      "F11 .req  v0\n" "qF11 .req  q0\n"
      "F12 .req  v1\n" "qF12 .req  q1\n"
      "F13 .req  v2\n" "qF13 .req  q2\n"
      "F14 .req  v3\n" "qF14 .req  q3\n"
      "F21 .req  v4\n" "qF21 .req  q4\n"
      "F22 .req  v5\n" "qF22 .req  q5\n"
      "F23 .req  v6\n" "qF23 .req  q6\n"
      "F24 .req  v7\n" "qF24 .req  q7\n"
      "F31 .req  v8\n" "qF31 .req  q8\n"
      "F32 .req  v9\n" "qF32 .req  q9\n"
      "F33 .req v10\n" "qF33 .req q10\n"
      "F34 .req v11\n" "qF34 .req q11\n"
      "F41 .req v12\n" "qF41 .req q12\n"
      "F42 .req v13\n" "qF42 .req q13\n"
      "F43 .req v14\n" "qF43 .req q14\n"
      "F44 .req v15\n" "qF44 .req q15\n"

      // Aliases for elements of the intermediate matrix `FZ`
      "FZ11 .req v16\n"
      "FZ12 .req v17\n"
      "FZ21 .req v18\n"
      "FZ22 .req v19\n"
      "FZ31 .req v20\n"
      "FZ32 .req v21\n"
      "FZ41 .req v22\n"
      "FZ42 .req v23\n"

      // Aliases for elements of the output matrix `f` (called `g` due to case
      // insensitivity of aliases).
      " g11 .req v24\n"
      "qg11 .req q24\n"
      " g12 .req v25\n"
      "qg12 .req q25\n"
      " g21 .req v26\n"
      "qg21 .req q26\n"
      " g22 .req v27\n"
      "qg22 .req q27\n"

      // Prepare the various strides
      "col_stride .req %x[col_stride]\n"
      "row_stride .req %x[row_stride]\n"
      "row_plus_col_stride .req %x[row_plus_col_stride]\n"

      "mstride1 .req %x[mstride1]\n"
      "mstride2 .req %x[mstride2]\n"
      "mstride3 .req %x[mstride3]\n"

      "tile_i  .req x19\n"  // Tile row counter
      "tile_j  .req x20\n"  // Tile column counter
      "channel .req x21\n"  // Channel counter

      "1:"  // Loop over batches
        "mov tile_i, %x[tile_M]\n"  // Reset tile row counter

        "2:"  // Loop over rows of tiles
          "mov tile_j, %x[tile_N]\n"  // Reset tile column counter

          "3:"  // Loop over columns of tiles
            // Perform initial loads of the matrix `F`
            "ldr qF11, [%x[inptr0]]\n"
            "ldr qF12, [%x[inptr0], mstride1]\n"
            "ldr qF13, [%x[inptr0], mstride2]\n"
            "ldr qF14, [%x[inptr0], mstride3]\n"
            "add %x[inptr0], %x[inptr0], #0x10\n"
            "ldr qF21, [%x[inptr4]]\n"
            "ldr qF22, [%x[inptr4], mstride1]\n"
            "subs channel, %x[n_channels], #4\n"  // Reset channel counter

            "ldr qF23, [%x[inptr4], mstride2]\n"
            "ldr qF24, [%x[inptr4], mstride3]\n"
            "add %x[inptr4], %x[inptr4], #0x10\n"
            "beq 5f\n"  // Jump straight to tail if necessary

            "4:"  // Loop over channels
              "ldr qF31, [%x[inptr8]]\n"
              "fadd FZ11.4s,  F11.4s, F12.4s\n"

              "ldr qF32, [%x[inptr8], mstride1]\n"
              "fsub FZ12.4s,  F12.4s, F13.4s\n"

              "ldr qF33, [%x[inptr8], mstride2]\n"
              "fadd FZ11.4s, FZ11.4s, F13.4s\n"

              "ldr qF34, [%x[inptr8], mstride3]\n"
              "fsub FZ12.4s, FZ12.4s, F14.4s\n"

              "ldr qF41, [%x[inptr12]]\n"
              "fadd FZ21.4s,  F21.4s, F22.4s\n"

              "ldr qF42, [%x[inptr12], mstride1]\n"
              "fsub FZ22.4s,  F22.4s, F23.4s\n"

              "ldr qF43, [%x[inptr12], mstride2]\n"
              "fadd FZ21.4s, FZ21.4s, F23.4s\n"

              "ldr qF44, [%x[inptr12], mstride3]\n"
              "fsub FZ22.4s, FZ22.4s, F24.4s\n"

              "fadd FZ31.4s,  F31.4s, F32.4s\n"
              "add %x[inptr8], %x[inptr8], #0x10\n"

              "fsub FZ32.4s,  F32.4s, F33.4s\n"
              "add %x[inptr12], %x[inptr12], #0x10\n"

              "fadd FZ31.4s, FZ31.4s, F33.4s\n"

              "fsub FZ32.4s, FZ32.4s, F34.4s\n"

              "fadd g11.4s, FZ11.4s, FZ21.4s\n"

              "fadd g12.4s, FZ12.4s, FZ22.4s\n"

              "fadd g11.4s,  g11.4s, FZ31.4s\n"

              "fadd g12.4s,  g12.4s, FZ32.4s\n"

              "ldr qF11, [%x[inptr0]]\n"
              "fadd FZ41.4s,  F41.4s, F42.4s\n"

              "ldr qF12, [%x[inptr0], mstride1]\n"
              "fsub g21.4s, FZ21.4s, FZ31.4s\n"

              "ldr qF13, [%x[inptr0], mstride2]\n"
              "fsub FZ42.4s,  F42.4s, F43.4s\n"

              "ldr qF14, [%x[inptr0], mstride3]\n"
              "str qg11, [%x[outptr]]\n"

              "ldr qF21, [%x[inptr4]]\n"
              "fadd FZ41.4s, FZ41.4s, F43.4s\n"

              "ldr qF22, [%x[inptr4], mstride1]\n"
              "str qg12, [%x[outptr], col_stride]\n"

              "ldr qF23, [%x[inptr4], mstride2]\n"
              "fsub FZ42.4s, FZ42.4s, F44.4s\n"

              "ldr qF24, [%x[inptr4], mstride3]\n"
              "fsub g22.4s, FZ22.4s, FZ32.4s\n"

              "fsub g21.4s,  g21.4s, FZ41.4s\n"
              "add %x[inptr0], %x[inptr0], #0x10\n"

              "fsub g22.4s,  g22.4s, FZ42.4s\n"
              "add %x[inptr4], %x[inptr4], #0x10\n"

              "subs channel, channel, #4\n"

              "str qg21, [%x[outptr], row_stride]\n"

              "str qg22, [%x[outptr], row_plus_col_stride]\n"

              "add %x[outptr], %x[outptr], #0x10\n"

              "bne 4b\n"

            "5:"  // Channel tail
              "ldr qF31, [%x[inptr8]]\n"
              "fadd FZ11.4s,  F11.4s, F12.4s\n"

              "ldr qF32, [%x[inptr8], mstride1]\n"
              "fsub FZ12.4s,  F12.4s, F13.4s\n"

              "ldr qF33, [%x[inptr8], mstride2]\n"
              "fadd FZ11.4s, FZ11.4s, F13.4s\n"

              "ldr qF34, [%x[inptr8], mstride3]\n"
              "fsub FZ12.4s, FZ12.4s, F14.4s\n"

              "ldr qF41, [%x[inptr12]]\n"
              "fadd FZ21.4s,  F21.4s, F22.4s\n"

              "ldr qF42, [%x[inptr12], mstride1]\n"
              "fsub FZ22.4s,  F22.4s, F23.4s\n"

              "ldr qF43, [%x[inptr12], mstride2]\n"
              "fadd FZ21.4s, FZ21.4s, F23.4s\n"

              "ldr qF44, [%x[inptr12], mstride3]\n"
              "fsub FZ22.4s, FZ22.4s, F24.4s\n"

              "fadd FZ31.4s,  F31.4s, F32.4s\n"
              "add %x[inptr8], %x[inptr8], #0x10\n"

              "fsub FZ32.4s,  F32.4s, F33.4s\n"
              "add %x[inptr12], %x[inptr12], #0x10\n"

              "fadd FZ31.4s, FZ31.4s, F33.4s\n"

              "fsub FZ32.4s, FZ32.4s, F34.4s\n"

              "fadd g11.4s, FZ11.4s, FZ21.4s\n"

              "fadd g12.4s, FZ12.4s, FZ22.4s\n"

              "fadd g11.4s,  g11.4s, FZ31.4s\n"

              "fadd g12.4s,  g12.4s, FZ32.4s\n"

              "fadd FZ41.4s,  F41.4s, F42.4s\n"

              "fsub g21.4s, FZ21.4s, FZ31.4s\n"

              "fsub FZ42.4s,  F42.4s, F43.4s\n"

              "str qg11, [%x[outptr]]\n"

              "fadd FZ41.4s, FZ41.4s, F43.4s\n"

              "str qg12, [%x[outptr], col_stride]\n"

              "fsub FZ42.4s, FZ42.4s, F44.4s\n"

              "fsub g22.4s, FZ22.4s, FZ32.4s\n"

              "fsub g21.4s,  g21.4s, FZ41.4s\n"

              "fsub g22.4s,  g22.4s, FZ42.4s\n"

              "subs channel, channel, #4\n"

              "str qg21, [%x[outptr], row_stride]\n"

              // Progress input pointers to the next row of the matrix
              "add  %x[inptr0],  %x[inptr0], %x[mrowpad]\n"
              "add  %x[inptr4],  %x[inptr4], %x[mrowpad]\n"
              "add  %x[inptr8],  %x[inptr8], %x[mrowpad]\n"
              "add %x[inptr12], %x[inptr12], %x[mrowpad]\n"

              "str qg22, [%x[outptr], row_plus_col_stride]\n"

              "add %x[outptr], %x[outptr], #0x10\n"


            "add %x[outptr], %x[outptr], col_stride\n"
            "subs tile_j, tile_j, #1\n"
            "bne 3b\n"

          "add %x[outptr], %x[outptr], row_stride\n"
          "subs tile_i, tile_i, #1\n"
          "bne 2b\n"

        "subs %w[batch], %w[batch], #1\n"
        "bne 1b\n"

      ".unreq  F11\n" ".unreq qF11\n"
      ".unreq  F12\n" ".unreq qF12\n"
      ".unreq  F13\n" ".unreq qF13\n"
      ".unreq  F14\n" ".unreq qF14\n"
      ".unreq  F21\n" ".unreq qF21\n"
      ".unreq  F22\n" ".unreq qF22\n"
      ".unreq  F23\n" ".unreq qF23\n"
      ".unreq  F24\n" ".unreq qF24\n"
      ".unreq  F31\n" ".unreq qF31\n"
      ".unreq  F32\n" ".unreq qF32\n"
      ".unreq  F33\n" ".unreq qF33\n"
      ".unreq  F34\n" ".unreq qF34\n"
      ".unreq  F41\n" ".unreq qF41\n"
      ".unreq  F42\n" ".unreq qF42\n"
      ".unreq  F43\n" ".unreq qF43\n"
      ".unreq  F44\n" ".unreq qF44\n"

      ".unreq FZ11\n" ".unreq FZ12\n"
      ".unreq FZ21\n" ".unreq FZ22\n"
      ".unreq FZ31\n" ".unreq FZ32\n"
      ".unreq FZ41\n" ".unreq FZ42\n"

      ".unreq  g11\n" ".unreq qg11\n"
      ".unreq  g12\n" ".unreq qg12\n"
      ".unreq  g21\n" ".unreq qg21\n"
      ".unreq  g22\n" ".unreq qg22\n"

      ".unreq col_stride\n"
      ".unreq row_stride\n"
      ".unreq row_plus_col_stride\n"

      ".unreq mstride1\n"
      ".unreq mstride2\n"
      ".unreq mstride3\n"

      ".unreq tile_i \n"
      ".unreq tile_j \n"
      ".unreq channel\n"

    : [batch] "+r" (batch),
      [outptr] "+r" (outptr),
      [inptr0] "+r" (inptr0),
      [inptr4] "+r" (inptr4),
      [inptr8] "+r" (inptr8),
      [inptr12] "+r" (inptr12)
    : [tile_M] "r" (tile_M),
      [tile_N] "r" (tile_N),
      [n_channels] "r" (output_shape.n_channels),
      [col_stride] "r" (col_stride),
      [row_stride] "r" (row_stride),
      [row_plus_col_stride] "r" (row_stride + col_stride),
      [mstride1] "r" (mstride * sizeof(float)),
      [mstride2] "r" (2 * mstride * sizeof(float)),
      [mstride3] "r" (3 * mstride * sizeof(float)),
      [mrowpad] "r" ((matrix_row_stride - output_shape.n_channels) * sizeof(float))
    : "x19", "x20", "x21",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24", "v25", "v26", "v27",
      "cc", "memory"
  );
}

template <>
template <bool tail_M, bool tail_N, const int channel_tail>
inline void Winograd2x2_3x3GemmOutput<float>::_execute(
    const Tensor4DShape &output_shape,
    float *output,
    const float *input,
    const int mstride,
    const int matrix_row_stride
) {
  // Compute basic information about the shape of the matrices
  const int tile_M = output_shape.n_rows / 2;
  const int tile_N = output_shape.n_cols / 2;
  const int n_channels = output_shape.n_channels;

  // Extract 16 input pointers
  const float* inptr[16];
  for (int i = 0; i < 16; i++) {
    inptr[i] = input + i*mstride;
  }

  // Extract 4 output pointers
  float *outptr00 = output;
  float *outptr01 = outptr00 + n_channels;
  float *outptr10 = outptr00 + output_shape.n_cols * n_channels;
  float *outptr11 = outptr10 + n_channels;

  // Progress over the output tiles, generating output values.
  for (int batch = 0; batch < output_shape.n_batches; batch++) {
    for (int tile_i = 0; tile_i < tile_M; tile_i++) {
      for (int tile_j = 0; tile_j < tile_N; tile_j++) {
        for (int channel = 0; channel < n_channels; channel++) {
          // Read values from the input pointers
          float F[4][4];
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              F[i][j] = *(inptr[i*4 + j]++);
            }
          }

          // Compute the matrix F.Z
          float ZF[4][2];
          ZF[0][0] = F[0][0] + F[0][1] + F[0][2];
          ZF[0][1] = F[0][1] - F[0][2] - F[0][3];
          ZF[1][0] = F[1][0] + F[1][1] + F[1][2];
          ZF[1][1] = F[1][1] - F[1][2] - F[1][3];
          ZF[2][0] = F[2][0] + F[2][1] + F[2][2];
          ZF[2][1] = F[2][1] - F[2][2] - F[2][3];
          ZF[3][0] = F[3][0] + F[3][1] + F[3][2];
          ZF[3][1] = F[3][1] - F[3][2] - F[3][3];

          // Hence compute the output matrix Z^T . (F.Z)
          *(outptr00++) = ZF[0][0] + ZF[1][0] + ZF[2][0];
          *(outptr01++) = ZF[0][1] + ZF[1][1] + ZF[2][1];
          *(outptr10++) = ZF[1][0] - ZF[2][0] - ZF[3][0];
          *(outptr11++) = ZF[1][1] - ZF[2][1] - ZF[3][1];
        }

        // Progress the input pointers to the next row
        for (int i = 0; i < 16; i++) {
          inptr[i] += matrix_row_stride - n_channels;
        }

        // Progress the output pointers to the next column
        outptr00 += n_channels;
        outptr01 += n_channels;
        outptr10 += n_channels;
        outptr11 += n_channels;
      }

      if (tail_N) {
        // Only evaluate the left-most columns of the output
        for (int channel = 0; channel < n_channels; channel++) {
          // Read values from the input pointers
          float F[4][3];
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
              F[i][j] = *(inptr[i*4 + j]++);
            }
          }
          for (int i = 0; i < 4; i++) {
            inptr[i*4 + 3]++;
          }

          // Compute the matrix F.Z
          float ZF[4][1];
          ZF[0][0] = F[0][0] + F[0][1] + F[0][2];
          ZF[1][0] = F[1][0] + F[1][1] + F[1][2];
          ZF[2][0] = F[2][0] + F[2][1] + F[2][2];
          ZF[3][0] = F[3][0] + F[3][1] + F[3][2];

          // Hence compute the output matrix Z^T . (F.Z)
          *(outptr00++) = ZF[0][0] + ZF[1][0] + ZF[2][0];
          *(outptr10++) = ZF[1][0] - ZF[2][0] - ZF[3][0];
        }

        // Progress the input pointers to the next row
        for (int i = 0; i < 16; i++) {
          inptr[i] += matrix_row_stride - n_channels;
        }

        // Progress the output pointers to the next column
        outptr01 += n_channels;  // Account for being skipped above
        outptr11 += n_channels;  // Account for being skipped above
      }

      // Progress the output pointers to the next row
      outptr00 += output_shape.n_cols * n_channels;
      outptr01 += output_shape.n_cols * n_channels;
      outptr10 += output_shape.n_cols * n_channels;
      outptr11 += output_shape.n_cols * n_channels;
    }

    if (tail_M) {
      // Only work on the upper row of the output
      for (int tile_j = 0; tile_j < tile_N; tile_j++) {
        for (int channel = 0; channel < n_channels; channel++) {
          // Read values from the input pointers
          float F[3][4];
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
              F[i][j] = *(inptr[i*4 + j]++);
            }
          }
          for (int j = 0; j < 4; j++) {
            inptr[12 + j]++;
          }

          // Compute the matrix F.Z
          float ZF[3][2];
          ZF[0][0] = F[0][0] + F[0][1] + F[0][2];
          ZF[0][1] = F[0][1] - F[0][2] - F[0][3];
          ZF[1][0] = F[1][0] + F[1][1] + F[1][2];
          ZF[1][1] = F[1][1] - F[1][2] - F[1][3];
          ZF[2][0] = F[2][0] + F[2][1] + F[2][2];
          ZF[2][1] = F[2][1] - F[2][2] - F[2][3];

          // Hence compute the output matrix Z^T . (F.Z)
          *(outptr00++) = ZF[0][0] + ZF[1][0] + ZF[2][0];
          *(outptr01++) = ZF[0][1] + ZF[1][1] + ZF[2][1];
        }

        // Progress the input pointers to the next row
        for (int i = 0; i < 16; i++) {
          inptr[i] += matrix_row_stride - n_channels;
        }

        // Progress the output pointers to the next column
        outptr00 += n_channels;
        outptr01 += n_channels;
        outptr10 += 2 * n_channels;  // Account for being skipped above
        outptr11 += 2 * n_channels;  // Account for being skipped above
      }

      if (tail_N) {
        // Only evaluate the upper-left cell of the output
        for (int channel = 0; channel < n_channels; channel++) {
          // Read values from the input pointers
          float F[3][3];
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              F[i][j] = *(inptr[i*4 + j]);
            }
          }
          for (int i = 0; i < 16; i++) {
            inptr[i]++;
          }

          // Compute the matrix F.Z
          float ZF[3][1];
          ZF[0][0] = F[0][0] + F[0][1] + F[0][2];
          ZF[1][0] = F[1][0] + F[1][1] + F[1][2];
          ZF[2][0] = F[2][0] + F[2][1] + F[2][2];

          // Hence compute the output matrix Z^T . (F.Z)
          *(outptr00++) = ZF[0][0] + ZF[1][0] + ZF[2][0];
        }

        // Progress the input pointers to the next row
        for (int i = 0; i < 16; i++) {
          inptr[i] += matrix_row_stride - n_channels;
        }

        // Progress the output pointers to the next column
        outptr01 += n_channels;  // Account for being skipped above
        outptr10 += n_channels;  // Account for being skipped above
        outptr11 += n_channels;  // Account for being skipped above
      }
    }
  }
}

/*****************************************************************************/
template <>
inline void Winograd2x2_3x3GemmOutput<float>::execute(
    const Tensor4DShape &output_shape,
    float* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    float* const output
) {
  // Dispatch to an appropriate implementation based on the shape of the output
  // tensor.
  if (output_shape.n_rows % 2 && output_shape.n_cols % 2) {
    constexpr bool tail_M = true, tail_N = true;
    switch (output_shape.n_channels % 4) {
      case 0:
        _execute<tail_M, tail_N, 0>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 1:
        _execute<tail_M, tail_N, 1>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 2:
        _execute<tail_M, tail_N, 2>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 3:
        _execute<tail_M, tail_N, 3>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      default:
        assert(0);
        break;
    }
  } else if (output_shape.n_rows % 2) {
    constexpr bool tail_M = true, tail_N = false;
    switch (output_shape.n_channels % 4) {
      case 0:
        _execute<tail_M, tail_N, 0>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 1:
        _execute<tail_M, tail_N, 1>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 2:
        _execute<tail_M, tail_N, 2>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 3:
        _execute<tail_M, tail_N, 3>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      default:
        assert(0);
        break;
    }
  } else if (output_shape.n_cols % 2) {
    constexpr bool tail_M = false, tail_N = true;
    switch (output_shape.n_channels % 4) {
      case 0:
        _execute<tail_M, tail_N, 0>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 1:
        _execute<tail_M, tail_N, 1>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 2:
        _execute<tail_M, tail_N, 2>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 3:
        _execute<tail_M, tail_N, 3>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      default:
        assert(0);
        break;

    }
  } else {
    constexpr bool tail_M = false, tail_N = false;
    switch (output_shape.n_channels % 4) {
      case 0:
        _execute<tail_M, tail_N, 0>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 1:
        _execute<tail_M, tail_N, 1>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 2:
        _execute<tail_M, tail_N, 2>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      case 3:
        _execute<tail_M, tail_N, 3>(output_shape, output, matrix_base, matrix_stride, matrix_row_stride);
        break;
      default:
        assert(0);
        break;

    }
  }
}
/*****************************************************************************/

}  // namespace winograd
#endif  // __aarch64__
