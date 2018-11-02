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
#include "../winograd_gemm.hpp"

namespace winograd
{
  /***************************************************************************/
  /* Instance-less API */
  template <int output_tile_rows, int output_tile_cols,
            int kernel_rows, int kernel_cols>
  template <typename T>
  void WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::InputTransform<T>::execute(
    const T* const input,        /** Input tensor data */
    const int n_batches,         /** Number of batches in input tensor. */
    const int in_batch_stride,   /** Stride between batches of the input. */
    const int n_rows,            /** Number of rows in input tensor. */
    const int in_row_stride,     /** Stride between rows of the input. */
    const int n_cols,            /** Number of columns in input tensor. */
    const int in_col_stride,     /** Stride between columns of the input. */
    const int n_channels,        /** Number of channels in input tensor. */
    const PaddingType padding,   /** Padding type. */
    const int tile_M,
    const int tile_N,
    T* const output,             /** Base of output matrices. */
    const int matrix_stride,     /** Stride between output matrices. */
    const int matrix_batch_stride,  /** Stride between batches within the matrix. */
    const int matrix_row_stride  /** Stride within matrices. */
  )
  {
    // Compute the padding required on each edge of the image
    const int pad_top = (padding == PADDING_SAME) ? (kernel_rows - 1) / 2 : 0;
    const int pad_left = (padding == PADDING_SAME) ? (kernel_cols - 1) / 2 : 0;
    const int tile_overlap = kernel_rows - 1;

    // Compute striding values (assuming NHWC ordered data)
    const int output_col_stride = matrix_row_stride;
    const int output_row_stride = tile_N * output_col_stride;

    // Loop over batches
    for (int batch = 0; batch < n_batches; batch++)
    {
      // Pointer to the batch
      const T* const input_base_batch = input + batch * in_batch_stride;
      T* const outptr_base_batch = output + batch * matrix_batch_stride;

      // Loop over rows of tiles
      for (int tile_i = 0; tile_i < tile_M; tile_i++)
      {
        // Pointer to the row
        const int row_offset = (tile_i == 0) ? 0 : pad_top;
        const T* const input_base_row = (
          input_base_batch + ((inner_tile_rows - (kernel_rows - 1))*tile_i - row_offset)*in_row_stride
        );
        T* const outptr_base_row = outptr_base_batch + tile_i*output_row_stride;

        // Padding (top + bottom) for the row
        const int row_top = tile_i*(inner_tile_rows - tile_overlap) - pad_top;
        const int row_bottom = row_top + inner_tile_rows;
        const int row_pad_top = (tile_i == 0) ? pad_top : 0;
        const int row_pad_bottom = (row_bottom <= n_rows) ? 0 : row_bottom - n_rows;

        // Process the row
        process_tile_row(
          tile_N, n_channels,
          input_base_row, in_row_stride, in_col_stride,
          outptr_base_row, matrix_stride, matrix_row_stride,
          row_pad_top, pad_left, row_pad_bottom, n_cols
        );
      }
    }
  }

  template <int output_tile_rows, int output_tile_cols,
            int kernel_rows, int kernel_cols>
  template <typename T>
  void WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::InputTransform<T>::process_tile_row(
    const int tile_N,
    int n_channels,
    const T* const input_base,
    const int input_row_stride,
    const int input_col_stride,
    T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    const int pad_top,
    const int row_pad_left,
    const int pad_bottom,
    const int n_cols
  )
  {
    constexpr int tile_overlap = kernel_cols - 1;

    // Loop over columns of tiles
    for (int tile_j = 0; tile_j < tile_N; tile_j++)
    {
      // Padding (left + right) for the tile
      const int t_pad_left = (tile_j == 0) ? row_pad_left : 0;
      const int t_start = tile_j*(inner_tile_cols - tile_overlap) - row_pad_left;
      const int t_end = t_start + inner_tile_cols;
      const int t_pad_right = (t_end <= n_cols) ? 0 : t_end - n_cols;

      // Get pointers into the inputs and outputs
      const int col_offset = (tile_j == 0) ? 0 : row_pad_left;
      const T* const input_base_col = (
        input_base + ((inner_tile_cols - tile_overlap)*tile_j - col_offset)*input_col_stride
      );
      T* const outptr = matrix_base + tile_j*matrix_row_stride;

      // Apply the specific tile processing function
      const int f_pad_top = pad_top ? 1 : 0;
      const int f_pad_left = t_pad_left ? 1 : 0;
      tile_fns[f_pad_top][f_pad_left][pad_bottom][t_pad_right](
        n_channels,
        input_base_col,
        input_row_stride,
        input_col_stride,
        outptr,
        matrix_stride
      );
    }
  }

  /***************************************************************************/
  template <int otr, int otc, int kr, int kc>
  template <typename T>
  WinogradGEMM<otr, otc, kr, kc>::InputTransform<T>::InputTransform(
    const T* const input,        /** Input tensor data */
    const int n_batches,         /** Number of batches in input tensor. */
    const int n_rows,            /** Number of rows in input tensor. */
    const int n_cols,            /** Number of columns in input tensor. */
    const int n_channels,        /** Number of channels in input tensor. */
    const PaddingType padding,   /** Padding type. */
    T* const output,             /** Base of output matrices. */
    const int matrix_stride,     /** Stride between output matrices. */
    const int matrix_row_stride, /** Stride within matrices. */
    const int in_batch_stride,   /** Stride between input batches. */
    const int in_row_stride,     /** Stride between input rows. */
    const int in_col_stride      /** Stride between input columns. */
  ) : _inptr(input), _outptr(output),
      _n_batches(n_batches), _n_rows(n_rows), _n_cols(n_cols), _n_channels(n_channels),
      _matrix_stride(matrix_stride), _matrix_row_stride(matrix_row_stride),
      _tiles_M(iceildiv((padding == PADDING_SAME) ? n_rows : n_rows - kr + 1,
                        output_tile_rows)),
      _tiles_N(iceildiv((padding == PADDING_SAME) ? n_cols : n_cols - kc + 1,
                        output_tile_cols)),
      _in_col_stride(in_col_stride ? in_col_stride : n_channels),
      _in_row_stride(in_row_stride ? in_row_stride : n_cols * _in_col_stride),
      _in_batch_stride(in_batch_stride ? in_batch_stride : n_rows * _in_row_stride),
      _padding_type(padding)
  {
  }

  template <int otr, int otc, int kr, int kc>
  template <typename T>
  unsigned int WinogradGEMM<otr, otc, kr, kc>::InputTransform<T>::get_window() const
  {
    // The final window includes the tail, all other windows will be a multiple
    // of the window block in size.
    return iceildiv(_n_channels, WINDOW_BLOCK);
  }

  template <int otr, int otc, int kr, int kc>
  template <typename T>
  void WinogradGEMM<otr, otc, kr, kc>::InputTransform<T>::run(
    const unsigned int start, const unsigned int stop
  )
  {
    if (start >= get_window())
    {
      return;
    }

    // Determine the window of work to perform
    const unsigned int start_channel = start * WINDOW_BLOCK;
    const unsigned int stop_channel = std::min<const unsigned int>(
      stop * WINDOW_BLOCK, _n_channels
    );
    const unsigned int n_channels = stop_channel - start_channel;

    // Perform the work
    execute(
      _inptr + start_channel,
      _n_batches, _in_batch_stride,
      _n_rows, _in_row_stride,
      _n_cols, _in_col_stride,
      n_channels,
      _padding_type,
      _tiles_M,
      _tiles_N,
      _outptr + start_channel,
      _matrix_stride,
      _matrix_row_stride * _tiles_M * _tiles_N,
      _matrix_row_stride
    );
  }
}
