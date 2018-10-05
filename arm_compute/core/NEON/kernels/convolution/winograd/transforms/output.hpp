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
  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  void OutputTransformImpl<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::execute(
    const int n_batches,
    const int output_batch_stride,
    const int n_rows,
    const int output_row_stride,
    const int n_cols,
    const int output_col_stride,
    const int n_channels,
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    const T* const biases,
    T* const output
  )
  {
    // Compute the number of tiles and hence the padding required on the bottom
    // and right of the image.
    const int tile_M = iceildiv(n_rows, OutputTileRows);
    const int tile_N = iceildiv(n_cols, OutputTileCols);
    const int pad_bottom = OutputTileRows*tile_M - n_rows;
    const int pad_right = OutputTileCols*tile_N - n_cols;

    const int matrix_tile_row_stride = tile_N * matrix_row_stride;
    const int matrix_batch_stride = tile_M * matrix_tile_row_stride;

    // Perform the output transformation for each batch
    for (int batch = 0; batch < n_batches; batch++)
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
        T* const outptr_row = outptr_batch + OutputTileRows*tile_i*output_row_stride;

        // Process the row
        process_tile_row(
          tile_N, n_channels, matrix_tile_row, matrix_stride,
          matrix_row_stride, biases,
          outptr_row, output_row_stride, output_col_stride, row_pad_bottom,
          pad_right
        );
      }
    }
  }

template <int KernelRows, int InnerTileRows, typename T>
  void OutputTransformImpl<KernelRows, 1, InnerTileRows, 1, T>::execute(
    const int n_batches,
    const int output_batch_stride,
    const int n_rows,
    const int output_row_stride,
    const int n_cols,
    const int output_col_stride,
    const int n_channels,
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    const T* const biases,
    T* const output
  )
  {
    // If an Nx1 kernel then transpose and redirect to the 1xN implementation.
    OutputTransformImpl<1, KernelRows, 1, InnerTileRows, T>::execute(
        n_batches,
        output_batch_stride,
        n_cols, output_col_stride,
        n_rows, output_row_stride,
        n_channels,
        matrix_base, matrix_stride, matrix_row_stride,
        biases, output
      );
  }

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  void OutputTransformImpl<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::process_tile_row(
    const int tile_N,
    const int n_channels,
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    const T* const biases,
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
      T* const outptr = output + OutputTileCols *tile_j*output_col_stride;

      // Perform the output transformation
      const typename Tiles::TileFn tilefn = Tiles::get_tile_specialization(row_pad_bottom, tile_pad_right);
      tilefn(
        n_channels, matrix_row, matrix_stride, biases,
        outptr, output_row_stride, output_col_stride,
        row_pad_bottom, tile_pad_right
      );
    }
  }

/***************************************************************************/
  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  OutputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::OutputTransform(
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    const T* const biases,
    T* const output,
    const int n_batches,
    const int n_rows,
    const int n_cols,
    const int n_channels,
    const int out_batch_stride,
    const int out_row_stride,
    const int out_col_stride
  ) : _matrix_base(matrix_base), _biases(biases),
      _matrix_stride(matrix_stride), _matrix_row_stride(matrix_row_stride),
      _outptr(output), _n_batches(n_batches), _n_rows(n_rows), _n_cols(n_cols),
      _n_channels(n_channels), _tile_M(iceildiv(n_rows, OutputTileRows)),
      _tile_N(iceildiv(n_cols, OutputTileCols)),
      _out_col_stride(out_col_stride ? out_col_stride : n_channels),
      _out_row_stride(out_row_stride ? out_row_stride : n_cols * _out_col_stride),
      _out_batch_stride(out_batch_stride ? out_batch_stride : n_rows * _out_row_stride)
  {
  }

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  unsigned int OutputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::get_window() const
  {
    // The final window includes the tail, all other windows will be a multiple
    // of the window block in size.
    return iceildiv(_n_channels, WINDOW_BLOCK);
  }

template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  void OutputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::run(
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

    execute(
      _n_batches,
      _out_batch_stride,
      _n_rows,
      _out_row_stride,
      _n_cols,
      _out_col_stride,
      n_channels,
      _matrix_base + start_channel,
      _matrix_stride,
      _matrix_row_stride,
      (_biases != nullptr) ? _biases + start_channel : nullptr,
      _outptr + start_channel
    );
  }

 template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  void OutputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::execute(
    const int n_batches,
    const int out_batch_stride,
    const int n_rows,
    const int out_row_stride,
    const int n_cols,
    const int out_col_stride,
    const int n_channels,
    const T* const matrix_base,
    const int matrix_stride,
    const int matrix_row_stride,
    const T* const biases,
    T* const output
  )
  {
    Transform::execute(
      n_batches, out_batch_stride,
      n_rows, out_row_stride,
      n_cols, out_col_stride, n_channels,
      matrix_base, matrix_stride, matrix_row_stride,
      biases, output
    );
  }

  template <int KernelCols, int InnerTileCols, typename T>
  typename OutputTransformImplTiles<1, KernelCols, 1, InnerTileCols, T>::TileFn
    OutputTransformImplTiles<1, KernelCols, 1, InnerTileCols, T>::
      get_tile_specialization(const int pad_bottom, const int pad_right)
  {
    (void) pad_bottom;

    if (!pad_right)
    {
      // No padding, return unpadded specialisation
      return tilefn_unpadded;
    }
    else
    {
      return tilefn_right_padded[pad_right - 1];
    }
  }

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  typename OutputTransformImplTiles<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::TileFn
    OutputTransformImplTiles<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::
      get_tile_specialization(const int pad_bottom, const int pad_right)
  {
    if (!(pad_bottom || pad_right))
    {
      // No padding, return unpadded specialisation
      return tilefn_unpadded;
    }
    else if (pad_bottom && !pad_right)
    {
      return tilefn_bottom_padded[pad_bottom - 1];
    }
    else if (!pad_bottom && pad_right)
    {
      return tilefn_right_padded[pad_right - 1];
    }
    else
    {
      return tilefn_generic;
    }
  }
}  // namespace winograd
