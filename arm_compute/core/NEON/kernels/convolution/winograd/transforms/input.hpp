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
  void InputTransformImpl<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::execute(
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
    const int pad_top = (padding == PADDING_SAME) ? (KernelRows - 1) / 2 : 0;
    const int pad_left = (padding == PADDING_SAME) ? (KernelCols - 1) / 2 : 0;

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
        // Padding (top + bottom) for the row
        const int row_top = tile_i*(InnerTileRows - overlap_rows) - pad_top;
        const int row_bottom = row_top + InnerTileRows;
        const int row_pad_top = std::max(0, pad_top - tile_i*(InnerTileRows - overlap_rows));
        const int row_pad_bottom = (row_bottom <= n_rows) ? 0 : row_bottom - n_rows;

        // Pointer to the row
        const int row_offset = std::min(0, row_pad_top - pad_top);
        const T* const input_base_row = (
          input_base_batch + ((InnerTileRows - overlap_rows)*tile_i + row_offset)*in_row_stride
        );
        T* const outptr_base_row = outptr_base_batch + tile_i*output_row_stride;

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


  template <int KernelRows, int InnerTileRows, typename T>
  void InputTransformImpl<KernelRows, 1, InnerTileRows, 1, T>::execute(
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
    // If an Nx1 kernel then transpose and redirect to the 1xN implementation
    InputTransformImpl<1, KernelRows, 1, InnerTileRows, T>::execute(
      input,
      n_batches, in_batch_stride,
      n_cols, in_col_stride,
      n_rows, in_row_stride,
      n_channels, padding,
      tile_N, tile_M,
      output, matrix_stride, matrix_batch_stride, matrix_row_stride
    );
  }

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  void InputTransformImpl<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::process_tile_row(
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
    // Loop over columns of tiles
    for (int tile_j = 0; tile_j < tile_N; tile_j++)
    {
      // Padding (left + right) for the tile
      const int t_start = tile_j*(InnerTileCols - overlap_cols) - row_pad_left;
      const int t_end = t_start + InnerTileCols;
      const int t_pad_left = std::max(0, row_pad_left - tile_j*(InnerTileCols - overlap_cols));
      const int t_pad_right = (t_end <= n_cols) ? 0 : t_end - n_cols;

      // Get pointers into the inputs and outputs
      const int col_offset = std::min(0, t_pad_left - row_pad_left);
      const T* const input_base_col = (
        input_base + ((InnerTileCols - overlap_cols)*tile_j + col_offset)*input_col_stride
      );
      T* const outptr = matrix_base + tile_j*matrix_row_stride;

      // Apply the specific tile processing function
      const typename Tiles::TileFn tilefn = Tiles::get_tile_specialization(
        pad_top, t_pad_left, pad_bottom, t_pad_right
      );

      tilefn(
        n_channels,
        input_base_col, input_row_stride, input_col_stride,
        outptr, matrix_stride,
        pad_top, t_pad_left, pad_bottom, t_pad_right
      );
    }
  }

  /***************************************************************************/
  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  InputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::InputTransform(
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
      _tiles_M(iceildiv((padding == PADDING_SAME) ? n_rows : n_rows - KernelRows + 1,
                        InnerTileRows - KernelRows + 1)),
      _tiles_N(iceildiv((padding == PADDING_SAME) ? n_cols : n_cols - KernelCols + 1,
                        InnerTileCols - KernelCols + 1)),
      _in_col_stride(in_col_stride ? in_col_stride : n_channels),
      _in_row_stride(in_row_stride ? in_row_stride : n_cols * _in_col_stride),
      _in_batch_stride(in_batch_stride ? in_batch_stride : n_rows * _in_row_stride),
      _padding_type(padding)
  {
  }

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  unsigned int InputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::get_window() const
  {
    // The final window includes the tail, all other windows will be a multiple
    // of the window block in size.
    return iceildiv(_n_channels, WINDOW_BLOCK);
  }

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  void InputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::run(
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

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  void InputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::execute(
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
    Transform::execute(
      input, n_batches, in_batch_stride, n_rows, in_row_stride, n_cols,
      in_col_stride, n_channels, padding, tile_M, tile_N, output,
      matrix_stride, matrix_batch_stride, matrix_row_stride
    );
  }

  template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
  typename InputTransformImplTiles<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::TileFn
    InputTransformImplTiles<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>::
      get_tile_specialization(
        const int pad_top,
        const int pad_left,
        const int pad_bottom,
        const int pad_right
      )
  {
    if (!(pad_top || pad_left || pad_bottom || pad_right))
    {
      // No padding, return unpadded specialisation
      return tilefn_unpadded;
    }
    else if (pad_top && !(pad_left || pad_bottom || pad_right))
    {
      // Top padding only
      const int index = (pad_top - min_pad_top) / (InnerTileRows - overlap_rows);
      return tilefn_top_padded[index];
    }
    else if (!(pad_top) && pad_left && !(pad_bottom || pad_right))
    {
      // Left padding only
      const int index = (pad_left - min_pad_left) / (InnerTileCols - overlap_cols);
      return tilefn_left_padded[index];
    }
    else if (!(pad_top || pad_left) && pad_bottom && !(pad_right))
    {
      // Bottom padding only
      return tilefn_bottom_padded[pad_bottom - 1];
    }
    else if (!(pad_top || pad_left || pad_bottom) && pad_right)
    {
      // Right padding only
      return tilefn_right_padded[pad_right - 1];
    }
    else
    {
      // Combination of paddings, return an unspecialised method
      return tilefn_generic;
    }
  }

  template <int KernelCols, int InnerTileCols, typename T>
  typename InputTransformImplTiles<1, KernelCols, 1, InnerTileCols, T>::TileFn
    InputTransformImplTiles<1, KernelCols, 1, InnerTileCols, T>::
      get_tile_specialization(
        const int pad_top,
        const int pad_left,
        const int pad_bottom,
        const int pad_right
      )
  {
    (void) pad_top;
    (void) pad_bottom;

    if (!(pad_left || pad_right))
    {
      // No padding, return unpadded specialisation
      return tilefn_unpadded;
    }
    else if (pad_left && !pad_right)
    {
      // Left padding only
      const int index = (pad_left - min_pad_left) / (InnerTileCols - overlap_cols);
      return tilefn_left_padded[index];
    }
    else if (!pad_left && pad_right)
    {
      // Right padding only
      return tilefn_right_padded[pad_right - 1];
    }
    else
    {
      // Combination of paddings, return an unspecialised method
      return tilefn_generic;
    }
  }
}


