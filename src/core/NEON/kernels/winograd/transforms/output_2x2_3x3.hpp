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
  /* Transform from the Winograd domain back to the spatial domain.
   */
  template <typename T>
  struct Winograd2x2_3x3GemmOutput {
    static void execute(
      const Tensor4DShape &output_shape,
      T* const matrix_base,
      const int matrix_stride,
      const int matrix_row_stride,
      T* const output
    );

    protected:
    /* Specialised implementation method. */
    template <bool tail_M, bool tail_N, int channel_tail>
    static void _execute(
      const Tensor4DShape &output_shape,
      T *output,
      const T *input,
      const int matrix_stride,
      const int matrix_row_stride
    );
  };

  /* Two-stage implementation of the transformation from the Winograd domain.
   *
   * First computes Z.F and then computes (Z.F).Z^T.
   */
  template <typename T>
  struct Winograd2x2_3x3GemmOutput_TwoStage {
    static void execute(
      const Tensor4DShape &output_shape,
      T* const matrix_base,
      const int matrix_stride,
      const int matrix_row_stride,
      T* const output
    );

    protected:
    template <int channel_tail>
    static void compute_zf(
      const int n_rows, const int n_channels,
      T* const zf, const T* const input[16]
    );

    template <bool tail_M, bool tail_N, int channel_tail>
    static void compute_zfzT(
      const Tensor4DShape &output_shape,
      T* const output, const T* const zf
    );
  };
}

#include "output_2x2_3x3/a64_float.hpp"
// #include "output_2x2_3x3/a64_float_two_stage.hpp"

/*****************************************************************************/
/*
template <typename T>
void winograd::Winograd2x2_3x3GemmOutput<T>::execute(
    const Tensor4DShape &output_shape,
    const int tile_M,
    const int tile_N,
    T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    T* const output
) {
  T* const antipadding = reinterpret_cast<T *>(malloc(sizeof(T) * output_shape.n_channels));

  // Get input pointers
  const T* inptrs[16];
  for (int i = 0; i < 16; i++) {
    inptrs[i] = matrices[i];
  }

  for (int batch = 0; batch < output_shape.n_batches; batch++) {
    for (int tile_i = 0; tile_i < tile_M; tile_i++) {
      for (int tile_j = 0; tile_j < tile_N; tile_j++) {
        // Get pointers for each of the 4 output cells required for this computation
        T* outptrs[4];
        for (int cell_i = 0, c = 0; cell_i < 2; cell_i++) {
          for (int cell_j = 0; cell_j < 2; cell_j++, c++) {
            const int i = tile_i*2 + cell_i;
            const int j = tile_j*2 + cell_j;

            if (i < output_shape.n_rows && j < output_shape.n_cols) {
              outptrs[c] = output + (
                  (batch*output_shape.n_rows + i) * output_shape.n_cols +
                j) * output_shape.n_channels;
            } else {
              outptrs[c] = antipadding;
            }
          }  // cell_j
        }  // cell_i

        for (int n = 0; n < output_shape.n_channels; n++) {
          // Read 16 values and progress pointers
          T v[16];
          for (int i = 0; i < 16; i++) {
            v[i] = *(inptrs[i]++);
          }

          // Compute output for 4 pixels
          *(outptrs[0]++) = v[ 0] + v[ 1] + v[ 2] +
                            v[ 4] + v[ 5] + v[ 6] +
                            v[ 8] + v[ 9] + v[10];
          *(outptrs[1]++) = v[ 1] - v[ 2] - v[ 3] +
                            v[ 5] - v[ 6] - v[ 7] +
                            v[ 9] - v[10] - v[11];
          *(outptrs[2]++) = v[ 4] + v[ 5] + v[ 6] -
                            v[ 8] - v[ 9] - v[10] -
                            v[12] - v[13] - v[14];
          *(outptrs[3]++) = v[ 5] - v[ 6] - v[ 7] -
                            v[ 9] + v[10] + v[11] -
                            v[13] + v[14] + v[15];
        }  // output_channel
      }  // tile_j
    }  // tile_i
  }  // batch

  free(antipadding);
}
*/

/*****************************************************************************/
/*
template <typename T>
void winograd::Winograd2x2_3x3GemmOutput_TwoStage<T>::execute(
    const Tensor4DShape &output_shape,
    T* const matrices[16], T* const output
) {
  // Allocate memory for the intermediate matrices
  const int tile_M = iceildiv(output_shape.n_rows, 2);
  const int tile_N = iceildiv(output_shape.n_cols, 2);
  const int n_rows = output_shape.n_batches * tile_M * tile_N;
  const int n_channels = output_shape.n_channels;
  T* matrices_zf = reinterpret_cast<T*>(
    calloc(8 * n_rows * n_channels, sizeof(T))
  );
  
  // Perform the first stage transform, computing ZF.
  // Specializations should dispatch to different methods based on tail size.
  compute_zf<0>(n_rows, n_channels, matrices_zf, matrices);
  
  // Perform the second stage transform, finishing Z F Z^T - variable dispatch
  // based on size of the output. Specialisations can also dispatch based on
  // the tail-size of the channel.
  if (output_shape.n_rows % 2 && output_shape.n_cols % 2) {
    compute_zfzT<true, true, 0>(output_shape, output, matrices_zf);
  } else if (output_shape.n_rows % 2) {
    compute_zfzT<true, false, 0>(output_shape, output, matrices_zf);
  } else if (output_shape.n_cols % 2) {
    compute_zfzT<false, true, 0>(output_shape, output, matrices_zf);
  } else {
    compute_zfzT<false, false, 0>(output_shape, output, matrices_zf);
  }

  free(reinterpret_cast<void*>(matrices_zf));
}

template <typename T>
template <int channel_tail>
void winograd::Winograd2x2_3x3GemmOutput_TwoStage<T>::compute_zf(
    const int n_rows, const int n_channels,
    T* output, const T* const input[16]
) {
  // Extract 8 output pointers
  T* outptr[8];
  for (int i = 0; i < 8; i++) {
    outptr[i] = output + i*n_rows*n_channels;
  }

  // Copy the 16 input pointers
  const T* inptr[16];
  for (int i = 0; i < 16; i++) {
    inptr[i] = input[i];
  }

  // For every row of the matrices
  for (int i = 0; i < n_rows; i++) {
    // For every channel
    for (int j = 0; j < n_channels; j++) {
      // Extract values from the input matrices
      T val[16];
      for (int n = 0; n < 16; n++) {
        val[n] = *(inptr[n]++);
      }

      // Compute output values
      *(outptr[0]++) = val[0] + val[1] + val[2];
      *(outptr[1]++) = val[1] - val[2] - val[3];
      *(outptr[2]++) = val[4] + val[5] + val[6];
      *(outptr[3]++) = val[5] - val[6] - val[7];
      *(outptr[4]++) = val[8] + val[9] + val[10];
      *(outptr[5]++) = val[9] - val[10] - val[11];
      *(outptr[6]++) = val[12] + val[13] + val[14];
      *(outptr[7]++) = val[13] - val[14] - val[15];
    }
  }
}

template <typename T>
template <bool tail_M, bool tail_N, int channel_tail>
void winograd::Winograd2x2_3x3GemmOutput_TwoStage<T>::compute_zfzT(
    const Tensor4DShape &output_shape,
    T* const output, const T* const input
) {
  // Sizing information
  const int tile_M = output_shape.n_rows / 2;
  const int tile_N = output_shape.n_cols / 2;

  const int n_rows = (output_shape.n_batches *
                      (tile_M + (tail_M ? 1 : 0)) *
                      (tile_N + (tail_N ? 1 : 0)));
  const int n_channels = output_shape.n_channels;

  // Extract 8 input pointers
  const T* inptr[8];
  for (int i = 0; i < 8; i++) {
    inptr[i] = input + i*n_rows*n_channels;
  }

  // Extract 4 output pointers
  T* outptr00 = output;
  T* outptr01 = outptr00 + n_channels;
  T* outptr10 = outptr00 + output_shape.n_cols * n_channels;
  T* outptr11 = outptr10 + n_channels;

  // Progress over the output tiles, generating output values.
  for (int batch = 0; batch < output_shape.n_batches; batch++) {
    for (int tile_i = 0; tile_i < tile_M; tile_i++) {
      for (int tile_j = 0; tile_j < tile_N; tile_j++) {
        for (int channel = 0; channel < n_channels; channel++) {
          // Read values from the input pointers
          T v[8];
          for (int i = 0; i < 8; i++) {
            v[i] = *(inptr[i]++);
          }

          // Compute the output values and progress the output pointers.
          *(outptr00++) = v[0] + v[2] + v[4];
          *(outptr01++) = v[1] + v[3] + v[5];
          *(outptr10++) = v[2] - v[4] - v[6];
          *(outptr11++) = v[3] - v[5] - v[7];
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
          T v[8];
          for (int i = 0; i < 4; i++) {
            v[i * 2] = *inptr[i * 2];
          }
          for (int i = 0; i < 8; i++) {
            inptr[i]++;
          }

          // Compute the output values and progress the output pointers.
          *(outptr00++) = v[0] + v[2] + v[4];
          *(outptr10++) = v[2] - v[4] - v[6];
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
          T v[8];
          for (int i = 0; i < 8; i++) {
            v[i] = *(inptr[i]++);
          }

          // Compute the output values and progress the output pointers.
          *(outptr00++) = v[0] + v[2] + v[4];
          *(outptr01++) = v[1] + v[3] + v[5];
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
          T v[8];
          for (int i = 0; i < 3; i++) {
            v[i * 2] = *inptr[i * 2];
          }
          for (int i = 0; i < 8; i++) {
            inptr[i]++;
          }

          // Compute the output values and progress the output pointers.
          *(outptr00++) = v[0] + v[2] + v[4];
        }

        // Progress the output pointers to the next column
        outptr01 += n_channels;  // Account for being skipped above
        outptr10 += n_channels;  // Account for being skipped above
        outptr11 += n_channels;  // Account for being skipped above
      }
    }
  }
}
*/
