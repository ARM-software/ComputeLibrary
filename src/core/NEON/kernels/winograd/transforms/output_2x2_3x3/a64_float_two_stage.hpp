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

/*****************************************************************************/
// Compute ZF specializations

template <>
template <>
inline void winograd::Winograd2x2_3x3GemmOutput_TwoStage<float>::compute_zf<0>(
    const int n_rows, const int n_channels,
    float* output, const float* const input[16]
) {
  // Make copies of some variables
  int row = n_rows;
  float* outptr = output;
  const float* inptr = input[0];

  // Perform the transformation
  asm volatile (
    // "inptr0 .req %x[inptr]\n"
    "inptr1 .req x0\n"
    "inptr2 .req x1\n"
    "inptr3 .req x2\n"
    "inptr4 .req x3\n"
    "inptr5 .req x4\n"
    "inptr6 .req x5\n"
    "inptr7 .req x6\n"
    "inptr8 .req x7\n"
    "inptr9 .req x8\n"
    "inptr10 .req x9\n"
    "inptr11 .req x10\n"
    "inptr12 .req x11\n"
    "inptr13 .req x12\n"
    "inptr14 .req x13\n"
    "inptr15 .req x14\n"

    // "outptr0 .req %x[outptr]\n"
    "outptr1 .req x15\n"
    "outptr2 .req x16\n"
    "outptr3 .req x17\n"
    "outptr4 .req x18\n"
    "outptr5 .req x19\n"
    "outptr6 .req x20\n"
    "outptr7 .req x21\n"

    // Compute additional pointers into the input and output matrices.
    "mstride .req x22\n"  // Matrix stride
    "mul mstride, %x[row], %x[n_channels]\n"
    "lsl mstride, mstride, #2\n"  // * sizeof(float)

    "add inptr1, %x[inptr], mstride\n"
    "add inptr2, %x[inptr], mstride, LSL #1\n"
    "add inptr3, inptr2, mstride\n"
    "add inptr4, inptr3, mstride\n"
    "add inptr5, inptr4, mstride\n"
    "add inptr6, inptr5, mstride\n"
    "add inptr7, inptr6, mstride\n"
    "add inptr8, inptr7, mstride\n"
    "add inptr9, inptr8, mstride\n"
    "add inptr10, inptr9, mstride\n"
    "add inptr11, inptr10, mstride\n"
    "add inptr12, inptr11, mstride\n"
    "add inptr13, inptr12, mstride\n"
    "add inptr14, inptr13, mstride\n"
    "add inptr15, inptr14, mstride\n"

    "add outptr1, %[outptr], mstride\n"
    "add outptr2, outptr1, mstride\n"
    "add outptr3, outptr2, mstride\n"
    "add outptr4, outptr3, mstride\n"
    "add outptr5, outptr4, mstride\n"
    "add outptr6, outptr5, mstride\n"
    "add outptr7, outptr6, mstride\n"

    ".unreq mstride\n"

    "column .req x22\n"  // Column loop counter

    "1:"  // Loop over rows
      "ldr q0, [%x[inptr]], #0x10\n"
      "ldr q1, [inptr1], #0x10\n"
      "ldr q2, [inptr2], #0x10\n"
      "ldr q3, [inptr3], #0x10\n"
      "ldr q4, [inptr4], #0x10\n"
      "ldr q5, [inptr5], #0x10\n"
      "ldr q6, [inptr6], #0x10\n"
      "ldr q7, [inptr7], #0x10\n"
      "subs column, %x[n_channels], #0x4\n"
      "beq 3f\n"

      "2:"  // Loop over columns
        "ldr q8, [inptr8], #0x10\n"
        "prfm pldl1keep, [%x[inptr], #196]\n"
        "fadd v16.4s, v0.4s, v1.4s\n"

        "ldr q9, [inptr9], #0x10\n"
        "prfm pldl1keep, [inptr1, #196]\n"
        "fsub v17.4s, v1.4s, v2.4s\n"

        "ldr q10, [inptr10], #0x10\n"
        "prfm pldl1keep, [inptr2, #196]\n"
        "fadd v16.4s, v16.4s, v2.4s\n"

        "ldr q11, [inptr11], #0x10\n"
        "prfm pldl1keep, [inptr3, #196]\n"
        "fsub v17.4s, v17.4s, v3.4s\n"

        "ldr q12, [inptr12], #0x10\n"
        "prfm pldl1keep, [inptr4, #196]\n"
        "str q16, [%x[outptr]], #0x10\n"

        "ldr q13, [inptr13], #0x10\n"
        "prfm pldl1keep, [inptr5, #196]\n"
        "str q17, [outptr1], #0x10\n"

        "ldr q14, [inptr14], #0x10\n"
        "prfm pldl1keep, [inptr6, #196]\n"
        "fadd v16.4s, v4.4s, v5.4s\n"

        "ldr q15, [inptr15], #0x10\n"
        "prfm pldl1keep, [inptr7, #196]\n"
        "fsub v17.4s, v5.4s, v6.4s\n"

        "ldr q0, [%x[inptr]], #0x10\n"
        "prfm pldl1keep, [inptr8, #196]\n"
        "fadd v16.4s, v16.4s, v6.4s\n"

        "ldr q1, [inptr1], #0x10\n"
        "prfm pldl1keep, [inptr9, #196]\n"
        "fsub v17.4s, v17.4s, v7.4s\n"

        "ldr q2, [inptr2], #0x10\n"
        "prfm pldl1keep, [inptr10, #196]\n"
        "str q16, [outptr2], #0x10\n"

        "ldr q3, [inptr3], #0x10\n"
        "prfm pldl1keep, [inptr11, #196]\n"
        "str q17, [outptr3], #0x10\n"

        "ldr q4, [inptr4], #0x10\n"
        "prfm pldl1keep, [inptr12, #196]\n"
        "fadd v16.4s, v8.4s, v9.4s\n"

        "ldr q5, [inptr5], #0x10\n"
        "prfm pldl1keep, [inptr13, #196]\n"
        "fsub v17.4s, v9.4s, v10.4s\n"

        "ldr q6, [inptr6], #0x10\n"
        "prfm pldl1keep, [inptr14, #196]\n"
        "fadd v16.4s, v16.4s, v10.4s\n"

        "ldr q7, [inptr7], #0x10\n"
        "prfm pldl1keep, [inptr15, #196]\n"
        "fsub v17.4s, v17.4s, v11.4s\n"

        "str q16, [outptr4], #0x10\n"
        "fadd v16.4s, v12.4s, v13.4s\n"
        "fsub v18.4s, v13.4s, v14.4s\n"

        "str q17, [outptr5], #0x10\n"
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fsub v18.4s, v18.4s, v15.4s\n"

        "str q16, [outptr6], #0x10\n"
        "subs column, column, #0x4\n"

        "str q18, [outptr7], #0x10\n"
        "bne 2b\n"

      "3:"  // Tail
        "ldr q8, [inptr8], #0x10\n"
        "prfm pldl1keep, [%x[inptr], #196]\n"
        "fadd v16.4s, v0.4s, v1.4s\n"

        "ldr q9, [inptr9], #0x10\n"
        "prfm pldl1keep, [inptr1, #196]\n"
        "fsub v17.4s, v1.4s, v2.4s\n"

        "ldr q10, [inptr10], #0x10\n"
        "prfm pldl1keep, [inptr2, #196]\n"
        "fadd v16.4s, v16.4s, v2.4s\n"

        "ldr q11, [inptr11], #0x10\n"
        "prfm pldl1keep, [inptr3, #196]\n"
        "fsub v17.4s, v17.4s, v3.4s\n"

        "ldr q12, [inptr12], #0x10\n"
        "prfm pldl1keep, [inptr4, #196]\n"
        "str q16, [%x[outptr]], #0x10\n"

        "ldr q13, [inptr13], #0x10\n"
        "prfm pldl1keep, [inptr5, #196]\n"
        "str q17, [outptr1], #0x10\n"

        "ldr q14, [inptr14], #0x10\n"
        "prfm pldl1keep, [inptr6, #196]\n"
        "fadd v16.4s, v4.4s, v5.4s\n"

        "ldr q15, [inptr15], #0x10\n"
        "prfm pldl1keep, [inptr7, #196]\n"
        "fsub v17.4s, v5.4s, v6.4s\n"

        "prfm pldl1keep, [inptr8, #196]\n"
        "prfm pldl1keep, [inptr9, #196]\n"
        "fadd v16.4s, v16.4s, v6.4s\n"

        "prfm pldl1keep, [inptr10, #196]\n"
        "prfm pldl1keep, [inptr11, #196]\n"
        "fsub v17.4s, v17.4s, v7.4s\n"

        "prfm pldl1keep, [inptr12, #196]\n"
        "prfm pldl1keep, [inptr13, #196]\n"
        "str q16, [outptr2], #0x10\n"

        "prfm pldl1keep, [inptr14, #196]\n"
        "prfm pldl1keep, [inptr15, #196]\n"
        "str q17, [outptr3], #0x10\n"

        "fadd v16.4s, v8.4s, v9.4s\n"
        "fsub v17.4s, v9.4s, v10.4s\n"

        "fadd v16.4s, v16.4s, v10.4s\n"
        "fsub v17.4s, v17.4s, v11.4s\n"

        "str q16, [outptr4], #0x10\n"
        "fadd v16.4s, v12.4s, v13.4s\n"
        "fsub v18.4s, v13.4s, v14.4s\n"

        "str q17, [outptr5], #0x10\n"
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fsub v18.4s, v18.4s, v15.4s\n"

        "str q16, [outptr6], #0x10\n"
        "str q18, [outptr7], #0x10\n"

      "subs %x[row], %x[row], #0x1\n"
      "bne 1b\n"

    ".unreq inptr1\n"
    ".unreq inptr2\n"
    ".unreq inptr3\n"
    ".unreq inptr4\n"
    ".unreq inptr5\n"
    ".unreq inptr6\n"
    ".unreq inptr7\n"
    ".unreq inptr8\n"
    ".unreq inptr9\n"
    ".unreq inptr10\n"
    ".unreq inptr11\n"
    ".unreq inptr12\n"
    ".unreq inptr13\n"
    ".unreq inptr14\n"
    ".unreq inptr15\n"
    ".unreq outptr1\n"
    ".unreq outptr2\n"
    ".unreq outptr3\n"
    ".unreq outptr4\n"
    ".unreq outptr5\n"
    ".unreq outptr6\n"
    ".unreq outptr7\n"

    : [row] "+r" (row),
      [inptr] "+r" (inptr),
      [outptr] "+r" (outptr)
    : [n_channels] "r" (n_channels),
      [sizeof_float] "i" (sizeof(float))
    : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
      "q12", "q13", "q14", "q15", "q16", "q17", "x0", "x1", "x2", "x3", "x4",
      "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
      "x16", "x17", "x18", "x19", "x20", "x21", "x22", "cc", "memory"
  );
}

/*****************************************************************************/
// Compute ZFZ^T specializations

template <>
template <>
inline void winograd::Winograd2x2_3x3GemmOutput_TwoStage<float>::compute_zfzT<false, false, 0>(
    const Tensor4DShape &output_shape,
    float* const output, const float* const input
) {
  const int tile_M = output_shape.n_rows / 2;
  const int tile_N = output_shape.n_cols / 2;
  int batch = output_shape.n_batches;
  float *outptr = output;
  const float *inptr = input;

  asm volatile (
    // Compute input pointers
    "inptr1 .req x0\n"
    "inptr2 .req x1\n"
    "inptr3 .req x2\n"
    "inptr4 .req x3\n"
    "inptr5 .req x4\n"
    "inptr6 .req x5\n"
    "inptr7 .req x6\n"
    "inptr8 .req x7\n"

    "mstride .req x8\n"
    "mul mstride, %x[tile_M], %x[tile_N]\n"
    "mul mstride, mstride, %x[n_channels]\n"
    "lsl mstride, mstride, #2\n"  // * sizeof(float)

    "add inptr1, %[inptr], mstride\n"
    "add inptr2, inptr1, mstride\n"
    "add inptr3, inptr2, mstride\n"
    "add inptr4, inptr3, mstride\n"
    "add inptr5, inptr4, mstride\n"
    "add inptr6, inptr5, mstride\n"
    "add inptr7, inptr6, mstride\n"
    "add inptr8, inptr7, mstride\n"

    ".unreq mstride\n"

    // Compute initial output pointers
    "outptr01 .req  x8\n"
    "outptr10 .req  x9\n"
    "outptr11 .req x10\n"

    "add outptr01, %x[outptr], %x[n_channels], LSL #2\n"
    "add outptr10, %x[outptr], %x[row_stride], LSL #2\n"
    "add outptr11,   outptr10, %x[n_channels], LSL #2\n"

    "tile_i  .req x11\n"
    "tile_j  .req x12\n"
    "channel .req x13\n"

    "1:"  // Loop over batches
      "mov tile_i, %x[tile_M]\n"

      "2:"  // Loop over rows of output tiles
        "mov tile_j, %x[tile_N]\n"

        "3:"  // Loop over columns of output tiles
          "ldr q0, [%x[inptr]], #0x10\n"
          "ldr q2, [inptr2], #0x10\n"
          "subs channel, %x[n_channels], #0x4\n"

          "ldr q1, [inptr1], #0x10\n"
          "ldr q3, [inptr3], #0x10\n"
          "beq 6f\n"

          "4:"
            "ldr q4, [inptr4], #0x10\n"
            "ldr q5, [inptr5], #0x10\n"
            "fadd v16.4s, v0.4s, v2.4s\n"

            "ldr q6, [inptr6], #0x10\n"
            "ldr q7, [inptr7], #0x10\n"
            "fadd v17.4s, v1.4s, v3.4s\n"

            "ldr q8, [%x[inptr]], #0x10\n"
            "ldr q10, [inptr2], #0x10\n"
            "fadd v16.4s, v16.4s, v4.4s\n"

            "ldr q9, [inptr1], #0x10\n"
            "ldr q11, [inptr3], #0x10\n"
            "fadd v17.4s, v17.4s, v5.4s\n"

            "str q16, [%x[outptr]], #0x10\n"
            "prfm pldl1strm, [%x[inptr], #196]\n"
            "fsub v18.4s, v2.4s, v4.4s\n"

            "str q17, [outptr01], #0x10\n"
            "prfm pldl1strm, [inptr2, #196]\n"
            "fsub v19.4s, v3.4s, v5.4s\n"

            "prfm pldl1strm, [inptr1, #196]\n"
            "prfm pldl1strm, [inptr3, #196]\n"
            "fsub v18.4s, v18.4s, v6.4s\n"

            "prfm pldl1strm, [inptr4, #196]\n"
            "prfm pldl1strm, [inptr5, #196]\n"
            "fsub v19.4s, v19.4s, v7.4s\n"

            "str q18, [outptr10], #0x10\n"
            "prfm pldl1strm, [inptr6, #196]\n"
            "prfm pldl1strm, [inptr7, #196]\n"

            "subs channel, channel, #0x4\n"

            "str q19, [outptr11], #0x10\n"
            "beq 6f\n"  // Branch to tail

            "ldr q12, [inptr4], #0x10\n"
            "ldr q13, [inptr5], #0x10\n"
            "fadd v16.4s, v8.4s, v10.4s\n"

            "ldr q14, [inptr6], #0x10\n"
            "ldr q15, [inptr7], #0x10\n"
            "fadd v17.4s, v9.4s, v11.4s\n"

            "ldr q0, [%x[inptr]], #0x10\n"
            "ldr q2, [inptr2], #0x10\n"
            "fadd v16.4s, v16.4s, v12.4s\n"

            "ldr q1, [inptr1], #0x10\n"
            "ldr q3, [inptr3], #0x10\n"
            "fadd v17.4s, v17.4s, v13.4s\n"

            "str q16, [%x[outptr]], #0x10\n"
            "prfm pldl1strm, [%x[inptr], #196]\n"
            "fsub v18.4s, v10.4s, v12.4s\n"

            "str q17, [outptr01], #0x10\n"
            "prfm pldl1strm, [inptr2, #196]\n"
            "fsub v19.4s, v11.4s, v13.4s\n"

            "prfm pldl1strm, [inptr1, #196]\n"
            "prfm pldl1strm, [inptr3, #196]\n"
            "fsub v18.4s, v18.4s, v14.4s\n"

            "prfm pldl1strm, [inptr4, #196]\n"
            "prfm pldl1strm, [inptr5, #196]\n"
            "fsub v19.4s, v19.4s, v15.4s\n"

            "str q18, [outptr10], #0x10\n"
            "prfm pldl1strm, [inptr6, #196]\n"
            "prfm pldl1strm, [inptr7, #196]\n"

            "subs channel, channel, #0x4\n"

            "str q19, [outptr11], #0x10\n"
            "bne 4b\n"  // Continue loop

          "5:"  // Tail
            "ldr q12, [inptr4], #0x10\n"
            "ldr q13, [inptr5], #0x10\n"
            "fadd v16.4s, v8.4s, v10.4s\n"

            "ldr q14, [inptr6], #0x10\n"
            "ldr q15, [inptr7], #0x10\n"
            "fadd v17.4s, v9.4s, v11.4s\n"

            "fadd v16.4s, v16.4s, v12.4s\n"

            "fadd v17.4s, v17.4s, v13.4s\n"

            "str q16, [%x[outptr]], #0x10\n"
            "fsub v18.4s, v10.4s, v12.4s\n"
            "fsub v19.4s, v11.4s, v13.4s\n"

            "str q17, [outptr01], #0x10\n"
            "fsub v18.4s, v18.4s, v14.4s\n"
            "fsub v19.4s, v19.4s, v15.4s\n"

            "str q18, [outptr10], #0x10\n"
            "str q19, [outptr11], #0x10\n"
            "b 7f\n"

          "6:"  // Tail
            "ldr q4, [inptr4], #0x10\n"
            "ldr q5, [inptr5], #0x10\n"
            "fadd v16.4s, v0.4s, v2.4s\n"

            "ldr q6, [inptr6], #0x10\n"
            "ldr q7, [inptr7], #0x10\n"
            "fadd v17.4s, v1.4s, v3.4s\n"

            "fadd v16.4s, v16.4s, v4.4s\n"

            "fadd v17.4s, v17.4s, v5.4s\n"

            "str q16, [%x[outptr]], #0x10\n"
            "fsub v18.4s, v2.4s, v4.4s\n"
            "fsub v19.4s, v3.4s, v5.4s\n"

            "str q17, [outptr01], #0x10\n"
            "fsub v18.4s, v18.4s, v6.4s\n"
            "fsub v19.4s, v19.4s, v7.4s\n"

            "str q18, [outptr10], #0x10\n"
            "str q19, [outptr11], #0x10\n"

          "7:"
            "add %x[outptr], %x[outptr], %x[n_channels], LSL #2\n"
            "add outptr01, outptr01, %x[n_channels], LSL #2\n"
            "add outptr10, outptr10, %x[n_channels], LSL #2\n"
            "add outptr11, outptr11, %x[n_channels], LSL #2\n"

            "subs tile_j, tile_j, #1\n"
            "bne 3b\n"

        // Progress the output pointers to the new row
        "add %x[outptr], %x[outptr], %x[row_stride], LSL #2\n"
        "add   outptr01,   outptr01, %x[row_stride], LSL #2\n"
        "add   outptr10,   outptr10, %x[row_stride], LSL #2\n"
        "add   outptr11,   outptr11, %x[row_stride], LSL #2\n"

        "subs tile_i, tile_i, #1\n"
        "bne 2b\n"

      "subs %[batch], %[batch], #1\n"
      "bne 1b\n"
      "5:"

    ".unreq inptr1\n"
    ".unreq inptr2\n"
    ".unreq inptr3\n"
    ".unreq inptr4\n"
    ".unreq inptr5\n"
    ".unreq inptr6\n"
    ".unreq inptr7\n"
    ".unreq inptr8\n"
    ".unreq outptr01\n"
    ".unreq outptr10\n"
    ".unreq outptr11\n"
    : [batch] "+r" (batch),
      [outptr] "+r" (outptr),
      [inptr] "+r" (inptr)
    : [tile_M] "r" (tile_M),
      [tile_N] "r" (tile_N),
      [n_channels] "r" (output_shape.n_channels),
      [row_stride] "r" (output_shape.n_cols * output_shape.n_channels)
    : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
      "x12", "x13", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
      "cc", "memory"
  );
}
/*****************************************************************************/

/*****************************************************************************/
template <>
inline void winograd::Winograd2x2_3x3GemmOutput_TwoStage<float>::execute(
    const Tensor4DShape &output_shape,
    float* const matrices[16], float* const output
) {
  // profiler prof;

  // Allocate memory for the intermediate matrices
  const int tile_M = iceildiv(output_shape.n_rows, 2);
  const int tile_N = iceildiv(output_shape.n_cols, 2);
  const int n_rows = output_shape.n_batches * tile_M * tile_N;
  const int n_channels = output_shape.n_channels;
  float* matrices_zf = reinterpret_cast<float*>(
    calloc(8 * n_rows * n_channels, sizeof(float))
  );
  
  // Perform the first stage transform, computing ZF.
  const auto f_compute_zf = [&] () {
    switch (n_channels % 4) {
      case 0:
        compute_zf<0>(n_rows, n_channels, matrices_zf, matrices);
        break;
      case 1:
        compute_zf<1>(n_rows, n_channels, matrices_zf, matrices);
        break;
      case 2:
        compute_zf<2>(n_rows, n_channels, matrices_zf, matrices);
        break;
      case 3:
        compute_zf<3>(n_rows, n_channels, matrices_zf, matrices);
    };
  };
  // prof("Compute ZF", f_compute_zf, 16 * n_rows * n_channels * sizeof(float), 0, 8 * n_rows * n_channels * sizeof(float));
  f_compute_zf();
  
  // Perform the second stage transform, finishing Z F Z^T - variable dispatch
  // based on size of the output and the channel tail.
  const auto f_compute_zfzT = [&] () {
    if (output_shape.n_rows % 2 && output_shape.n_cols % 2) {
      constexpr bool tail_M = true, tail_N = true;
      switch (n_channels % 4) {
        case 0:
          compute_zfzT<tail_M, tail_N, 0>(output_shape, output, matrices_zf);
          break;
        case 1:
          compute_zfzT<tail_M, tail_N, 1>(output_shape, output, matrices_zf);
          break;
        case 2:
          compute_zfzT<tail_M, tail_N, 2>(output_shape, output, matrices_zf);
          break;
        case 3:
          compute_zfzT<tail_M, tail_N, 3>(output_shape, output, matrices_zf);
      }
    } else if (output_shape.n_rows % 2) {
      constexpr bool tail_M = true, tail_N = false;
      switch (n_channels % 4) {
        case 0:
          compute_zfzT<tail_M, tail_N, 0>(output_shape, output, matrices_zf);
          break;
        case 1:
          compute_zfzT<tail_M, tail_N, 1>(output_shape, output, matrices_zf);
          break;
        case 2:
          compute_zfzT<tail_M, tail_N, 2>(output_shape, output, matrices_zf);
          break;
        case 3:
          compute_zfzT<tail_M, tail_N, 3>(output_shape, output, matrices_zf);
      }
    } else if (output_shape.n_cols % 2) {
      constexpr bool tail_M = false, tail_N = true;
      switch (n_channels % 4) {
        case 0:
          compute_zfzT<tail_M, tail_N, 0>(output_shape, output, matrices_zf);
          break;
        case 1:
          compute_zfzT<tail_M, tail_N, 1>(output_shape, output, matrices_zf);
          break;
        case 2:
          compute_zfzT<tail_M, tail_N, 2>(output_shape, output, matrices_zf);
          break;
        case 3:
          compute_zfzT<tail_M, tail_N, 3>(output_shape, output, matrices_zf);
      }
    } else {
      constexpr bool tail_M = false, tail_N = false;
      switch (n_channels % 4) {
        case 0:
          compute_zfzT<tail_M, tail_N, 0>(output_shape, output, matrices_zf);
          break;
        case 1:
          compute_zfzT<tail_M, tail_N, 1>(output_shape, output, matrices_zf);
          break;
        case 2:
          compute_zfzT<tail_M, tail_N, 2>(output_shape, output, matrices_zf);
          break;
        case 3:
          compute_zfzT<tail_M, tail_N, 3>(output_shape, output, matrices_zf);
      }
    }
  };
  // prof("Compute ZFZT", f_compute_zfzT, 8 * n_rows * n_channels * sizeof(float), 0, 4 * n_rows * n_channels * sizeof(float));
  f_compute_zfzT();

  free(reinterpret_cast<void*>(matrices_zf));
}
/*****************************************************************************/

#endif  // __aarch64__
