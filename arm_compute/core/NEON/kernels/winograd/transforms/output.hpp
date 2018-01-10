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
  template <int output_tile_rows, int output_tile_cols,
            int kernel_rows, int kernel_cols>
  template <typename T>
  void WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::OutputTransform<T>::execute(
    const Tensor4DShape &output_shape,
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    T* const output
  )
  {
    // Compute the number of tiles and hence the padding required on the bottom
    // and right of the image.
    const int tile_M = iceildiv(output_shape.n_rows, output_tile_rows);
    const int tile_N = iceildiv(output_shape.n_cols, output_tile_cols);
    const int pad_bottom = output_tile_rows*tile_M - output_shape.n_rows;
    const int pad_right = output_tile_cols*tile_N - output_shape.n_cols;

    const int matrix_tile_row_stride = tile_N * matrix_row_stride;
    const int matrix_batch_stride = tile_M * matrix_tile_row_stride;
    const int output_col_stride = output_shape.n_channels;
    const int output_row_stride = output_shape.n_cols * output_col_stride;
    const int output_batch_stride = output_shape.n_rows * output_row_stride;

    // Perform the output transformation for each batch
    for (int batch = 0; batch < output_shape.n_batches; batch++)
    {
      // Get batch offset for input and outputs.
      const T* const matrix_batch = matrix_base + batch*matrix_batch_stride;
      T* const outptr_batch = output + batch*output_batch_stride;

      // Perform the output transformation for each row of the output tensor.
      for (int tile_i = 0; tile_i < tile_M; tile_i++)
      {
        // Compute properties of this row of output tiles
        const int row_pad_bottom = (tile_i < tile_M - 1) ? 0: pad_bottom;
        const T* const matrix_tile_row = matrix_batch + tile_i * matrix_tile_row_stride;
        T* const outptr_row = outptr_batch + output_tile_rows*tile_i*output_row_stride;

        // Process the row
        process_tile_row(
          tile_N, output_shape.n_channels, matrix_tile_row, matrix_stride,
          matrix_row_stride, outptr_row, output_row_stride,
          output_col_stride, row_pad_bottom, pad_right
        );
      }
    }
  }

  template <int output_tile_rows, int output_tile_cols,
            int kernel_rows, int kernel_cols>
  template <typename T>
  void WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::OutputTransform<T>::process_tile_row(
    const int tile_N,
    const int n_channels,
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    T* const output,
    const int output_row_stride,
    const int output_col_stride,
    const int row_pad_bottom,
    const int row_pad_right
  )
  {
    // Loop over columns of tiles
    for (int tile_j = 0; tile_j < tile_N; tile_j++)
    {
      // Properties of this tile
      const int tile_pad_right = (tile_j < tile_N - 1) ? 0 : row_pad_right;
      const T* const matrix_row = matrix_base + tile_j * matrix_row_stride;
      T* const outptr = output + output_tile_cols*tile_j*output_col_stride;

      // Perform the output transformation
      tile_fns[row_pad_bottom][tile_pad_right](
        n_channels, matrix_row, matrix_stride,
        outptr, output_row_stride, output_col_stride
      );
    }
  }

  template <int output_tile_rows, int output_tile_cols, int kr, int kc>
  template <typename T>
  size_t WinogradGEMM<output_tile_rows, output_tile_cols, kr, kc>::OutputTransform<T>::bytes_read(const Tensor4DShape &shape)
  {
    const int M = iceildiv(shape.n_rows, output_tile_rows) *
                  iceildiv(shape.n_cols, output_tile_cols);
    const int N = shape.n_channels;
    return inner_tile_rows * inner_tile_cols * M * N * sizeof(T);
  }

  template <int otr, int otc, int kr, int kc>
  template <typename T>
  size_t WinogradGEMM<otr, otc, kr, kc>::OutputTransform<T>::bytes_written(const Tensor4DShape &shape)
  {
    return shape.size() * sizeof(T);
  }

  template <int output_tile_rows, int output_tile_cols, int kr, int kc>
  template <typename T>
  WinogradGEMM<output_tile_rows, output_tile_cols, kr, kc>::OutputTransform<T>::OutputTransform(
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    T* const output,
    const int n_batches,
    const int n_rows,
    const int n_cols,
    const int n_channels
  ) : _matrix_base(matrix_base), _matrix_stride(matrix_stride), _matrix_row_stride(matrix_row_stride),
      _outptr(output), _n_batches(n_batches), _n_rows(n_rows), _n_cols(n_cols), _n_channels(n_channels),
      _tile_M(iceildiv(n_rows, output_tile_rows)), _tile_N(iceildiv(n_cols, output_tile_cols))
  {
  }

  template <int otr, int otc, int kr, int kc>
  template <typename T>
  unsigned int WinogradGEMM<otr, otc, kr, kc>::OutputTransform<T>::get_window() const
  {
    // TODO When the output transform supports multithreading, return the total
    // number of tile rows (allowing for multiple batches). For now we return 1
    // to indicate that the activations must be transformed as a single block.
    return 1;  // TODO _tile_M * _n_batches;
  }

  template <int otr, int otc, int kr, int kc>
  template <typename T>
  void WinogradGEMM<otr, otc, kr, kc>::OutputTransform<T>::run(
    const unsigned int start, const unsigned int stop
  )
  {
    // TODO When the output transform supports multithreading call execute for a
    // portion of the tile rows.
    (void) start;
    (void) stop;

    // For now, just do all of the work.
    const Tensor4DShape output_shape = {
      _n_batches, _n_rows, _n_cols, _n_channels, NHWC
    };
    execute(
      output_shape, _matrix_base, _matrix_stride, _matrix_row_stride, _outptr
    );
  }
}  // namespace winograd
