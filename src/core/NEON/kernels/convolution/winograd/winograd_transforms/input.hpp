/*
 * Copyright (c) 2019 ARM Limited.
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

#include "winograd.hpp"
#include "padding.hpp"

#define MEMBERFN(RTYPE) template <\
  int InnerTileRows, int InnerTileCols,\
  typename TIn, typename TOut, WinogradRoots Roots\
> RTYPE InputTransform<InnerTileRows, InnerTileCols, TIn, TOut, Roots>


#define Nx1MEMBERFN(RTYPE) template <\
  int InnerTileRows, typename TIn, typename TOut, WinogradRoots Roots\
> RTYPE InputTransform<InnerTileRows, 1, TIn, TOut, Roots>

namespace winograd
{

MEMBERFN()::InputTransform(
  const int kernel_rows,
  const int kernel_cols,
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels,
  const int padding_top,
  const int padding_left,
  const int padding_bottom,
  const int padding_right
) : _n_batches(n_batches), _n_rows(n_rows), _n_cols(n_cols), _n_channels(n_channels),
    _inptr(nullptr), _outptr(nullptr),
    _overlap_rows(kernel_rows - 1), _overlap_cols(kernel_cols - 1),
    _padding_top(padding_top), _padding_left(padding_left), _padding_bottom(padding_bottom), _padding_right(padding_right),
    _tiles_M(iceildiv(padding_top + n_rows + padding_bottom - kernel_rows + 1, InnerTileRows - kernel_rows + 1)),
    _tiles_N(iceildiv(padding_left + n_cols + padding_right - kernel_cols + 1, InnerTileCols - kernel_cols + 1)),
    _matrix_stride(0), _matrix_row_stride(0), _matrix_batch_stride(0),
    _in_col_stride(0), _in_row_stride(0), _in_batch_stride(0),
    _working_space_col_stride(n_channels),
    _working_space_row_stride(InnerTileCols * _working_space_col_stride),
    _working_space(nullptr)
{
}

MEMBERFN(void)::set_input_tensor(const void* const inptr)
{
  set_input_tensor(inptr, _n_channels);
}

MEMBERFN(void)::set_input_tensor(const void* const inptr, const int ldcol)
{
  set_input_tensor(inptr, _n_cols * ldcol, ldcol);
}

MEMBERFN(void)::set_input_tensor(const void* const inptr, const int ldrow, const int ldcol)
{
  set_input_tensor(inptr, _n_rows * ldrow, ldrow, ldcol);
}

MEMBERFN(void)::set_input_tensor(const void* const inptr, const int ldbatch, const int ldrow, const int ldcol)
{
  _inptr = static_cast<const TIn *>(inptr);
  _in_batch_stride = ldbatch;
  _in_row_stride = ldrow;
  _in_col_stride = ldcol;
}

MEMBERFN(void)::set_output_matrices(void * const mptr, const int ldmatrix, const int ldrow)
{
  _outptr = static_cast<TOut *>(mptr);
  _matrix_stride = ldmatrix;
  _matrix_row_stride = ldrow;
  _matrix_batch_stride = _tiles_M * _tiles_N * ldrow;
}

Nx1MEMBERFN()::InputTransform(
  const int kernel_rows,
  const int kernel_cols,
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels,
  const int padding_top,
  const int padding_left,
  const int padding_bottom,
  const int padding_right
) : InputTransform<1, InnerTileRows, TIn, TOut, Roots>::InputTransform(
    /* Transpose rows and columns */
    kernel_cols, kernel_rows, n_batches, n_cols, n_rows, n_channels,
    padding_left, padding_top, padding_right, padding_bottom
  )
{
}

Nx1MEMBERFN(void)::set_input_tensor(const void* const inptr)
{
  set_input_tensor(inptr, this->_n_channels);
}

Nx1MEMBERFN(void)::set_input_tensor(const void* const inptr, const int ldcol)
{
  set_input_tensor(inptr, this->_n_cols * ldcol, ldcol);
}

Nx1MEMBERFN(void)::set_input_tensor(const void* const inptr, const int ldrow, const int ldcol)
{
  set_input_tensor(inptr, this->_n_rows * ldrow, ldrow, ldcol);
}

Nx1MEMBERFN(void)::set_input_tensor(const void* const inptr, const int ldbatch, const int ldrow, const int ldcol)
{
  // Transpose row and column strides
  Base::set_input_tensor(inptr, ldbatch, ldcol, ldrow);
}

MEMBERFN(size_t)::get_working_space_size(const unsigned int nthreads) const
{
  return sizeof(TIn) * InnerTileRows * _working_space_row_stride * nthreads;
}

MEMBERFN(void)::set_working_space(void * const buffer)
{
  _working_space = static_cast<TIn *>(buffer);
}

MEMBERFN(unsigned int)::get_window(void) const
{
  return iceildiv(_n_channels, WINDOW_BLOCK);
}

MEMBERFN(void)::run(
  const unsigned int start,
  const unsigned int stop,
  const unsigned int threadid
)
{
  // Determine the channels on which to work
  if (start >= get_window())
  {
    return;  // No work to do beyond the end of the window
  }
  const unsigned int start_channel = start * WINDOW_BLOCK;
  const unsigned int stop_channel = std::min<unsigned int>(_n_channels , stop * WINDOW_BLOCK);
  const unsigned int n_channels = stop_channel - start_channel;

  // Loop over batches
  for (int batch = 0; batch < _n_batches; batch++)
  {
    const TIn* const inptr_batch = _inptr + start_channel + batch*_in_batch_stride;
    TOut* const outptr_batch = _outptr + start_channel + batch*_matrix_batch_stride;

    // Loop over rows of tiles
    for (int tile_i = 0; tile_i < _tiles_M; tile_i++)
    {
      // Compute the starting and ending row of pixels within the row of tiles,
      // hence compute the padding to apply to the top and bottom of each tile.
      const int row_top = tile_i * (InnerTileRows - _overlap_rows) - _padding_top;
      const int row_bottom = row_top + InnerTileRows;
      const int row_pad_top = std::max(0, _padding_top - tile_i * (InnerTileRows - _overlap_rows));
      const int row_pad_bottom = std::max(0, row_bottom - _n_rows);

      // Get a pointer to the start of the row.
      const int row_offset = std::min(0, row_pad_top - _padding_top);
      const TIn* const inptr_row = inptr_batch + _in_row_stride*(row_offset + tile_i*(InnerTileRows - _overlap_rows));
      TOut* const outptr_row = outptr_batch + tile_i*_tiles_N*_matrix_row_stride;

      // Loop over tiles within the row
      for (int tile_j = 0; tile_j < _tiles_N; tile_j++)
      {
        // Compute the starting and ending column of pixels within the tile,
        // hence compute the padding to apply to the left and right of the
        // tile.
        const int tile_left = tile_j * (InnerTileCols - _overlap_cols) - _padding_left;
        const int tile_right = tile_left + InnerTileCols;
        const int tile_pad_left = std::max(0, _padding_left - tile_j * (InnerTileCols - _overlap_cols));
        const int tile_pad_right = std::max(0, tile_right - _n_cols);

        // Get a pointer to the start of the tile.
        const int col_offset = std::min(0, tile_pad_left - _padding_left);
        const TIn* const inptr_tile = inptr_row + _in_col_stride*(col_offset + tile_j*(InnerTileCols - _overlap_cols));
        TOut* const outptr_tile = outptr_row + tile_j * _matrix_row_stride;

        // Transform the tile, applying padding if necessary.
        if (row_pad_top || tile_pad_left || row_pad_bottom || tile_pad_right)
        {
          transform_padded_tile(
            threadid, n_channels, outptr_tile, inptr_tile,
            row_pad_top, tile_pad_left, row_pad_bottom, tile_pad_right
          );
        }
        else
        {
          transform_unpadded_tile(threadid, n_channels, outptr_tile, inptr_tile);
        }
      }
    }
  }
}

MEMBERFN(void)::transform_unpadded_tile(
  const unsigned int /* threadid unused */,
  const int n_channels,
  TOut * const outptr,
  const TIn * const inptr
)
{
  transform_tile(
    n_channels, inptr, _in_row_stride, _in_col_stride, outptr, _matrix_stride
  );
}

MEMBERFN(void)::transform_padded_tile(
  const unsigned int threadid,
  const int n_channels,
  TOut * const outptr,
  const TIn * const inptr,
  const int padding_top,
  const int padding_left,
  const int padding_bottom,
  const int padding_right
)
{
  padding::copy_and_pad_tile(
    InnerTileRows, InnerTileCols, n_channels,
    inptr, _in_row_stride, _in_col_stride,
    static_cast<TIn *>(get_working_space(threadid)), _working_space_row_stride, _working_space_col_stride,
    padding_top, padding_left, padding_bottom, padding_right
  );

  transform_tile(
    n_channels, static_cast<const TIn *>(get_working_space(threadid)),
    _working_space_row_stride, _working_space_col_stride,
    outptr, _matrix_stride
  );
}

MEMBERFN(void *)::get_working_space(const unsigned int threadid) const
{
  return _working_space + InnerTileRows * _working_space_row_stride * threadid;
}

}  // namespace winograd
