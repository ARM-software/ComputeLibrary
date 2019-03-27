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

#include <algorithm>
#include "winograd.hpp"
#include "padding.hpp"
#include "utils.hpp"

#define MEMBERFN(RTYPE) template<\
  int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols,\
  typename TIn, typename TOut, WinogradRoots Roots\
> RTYPE OutputTransform<KernelRows, KernelCols, InnerTileRows, InnerTileCols, TIn, TOut, Roots>

#define Nx1MEMBERFN(RTYPE) template<\
  int KernelRows, int InnerTileRows, typename TIn, typename TOut, WinogradRoots Roots\
> RTYPE OutputTransform<KernelRows, 1, InnerTileRows, 1, TIn, TOut, Roots>

namespace winograd
{

MEMBERFN()::OutputTransform(
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels
) : _n_batches(n_batches), _n_rows(n_rows), _n_cols(n_cols), _n_channels(n_channels),
    _matrix_base(nullptr),
    _biases(nullptr),
    _matrix_stride(0), _matrix_row_stride(0), _matrix_batch_stride(0),
    _outptr(nullptr),
    _tiles_M(iceildiv(n_rows, output_tile_rows)),
    _tiles_N(iceildiv(n_cols, output_tile_cols)),
    _out_col_stride(0), _out_row_stride(0), _out_batch_stride(0),
    _working_space_col_stride(n_channels),
    _working_space_row_stride(output_tile_cols * _working_space_col_stride),
    _working_space(nullptr)
{
}

MEMBERFN(void)::set_input_matrices(const void * const mptr, const int ldmatrix, const int ldrow)
{
  _matrix_base = static_cast<const TIn *>(mptr);
  _matrix_stride = ldmatrix;
  _matrix_row_stride = ldrow;
  _matrix_batch_stride = _tiles_M * _tiles_N * ldrow;
}

MEMBERFN(void)::set_bias(const void * const bias)
{
  _biases = static_cast<const TOut *>(bias);
}

MEMBERFN(void)::set_output_tensor(void * const outptr)
{
  set_output_tensor(outptr, _n_channels);
}

MEMBERFN(void)::set_output_tensor(void * const outptr, const int ldcol)
{
  set_output_tensor(outptr, _n_cols * ldcol, ldcol);
}

MEMBERFN(void)::set_output_tensor(void * const outptr, const int ldrow, const int ldcol)
{
  set_output_tensor(outptr, _n_rows * ldrow, ldrow, ldcol);
}

MEMBERFN(void)::set_output_tensor(void * const outptr, const int ldbatch, const int ldrow, const int ldcol)
{
  _outptr = static_cast<TOut *>(outptr);
  _out_batch_stride = ldbatch;
  _out_row_stride = ldrow;
  _out_col_stride = ldcol;
}

Nx1MEMBERFN()::OutputTransform(
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels
) : OutputTransform<1, KernelRows, 1, InnerTileRows, TIn, TOut, Roots>::OutputTransform(
    n_batches, n_cols, n_rows, n_channels /* Transpose rows and columns */
  )
{
}

Nx1MEMBERFN(void)::set_output_tensor(void * const outptr)
{
  set_output_tensor(outptr, this->_n_channels);
}

Nx1MEMBERFN(void)::set_output_tensor(void * const outptr, const int ldcol)
{
  set_output_tensor(outptr, this->_n_cols * ldcol, ldcol);
}

Nx1MEMBERFN(void)::set_output_tensor(void * const outptr, const int ldrow, const int ldcol)
{
  set_output_tensor(outptr, this->_n_rows * ldrow, ldrow, ldcol);
}

Nx1MEMBERFN(void)::set_output_tensor(void * const outptr, const int ldbatch, const int ldrow, const int ldcol)
{
  // Transpose rows and columns
  Base::set_output_tensor(outptr, ldbatch, ldcol, ldrow);
}

MEMBERFN(size_t)::get_working_space_size(const unsigned int nthreads) const
{
  return sizeof(TOut) * output_tile_rows * _working_space_row_stride * nthreads;
}

MEMBERFN(void)::set_working_space(void * const buffer)
{
  _working_space = static_cast<TOut *>(buffer);
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
  const unsigned int stop_channel = std::min<unsigned int>(_n_channels, stop * WINDOW_BLOCK);
  const unsigned int n_channels = stop_channel - start_channel;

  const auto matrix_tile_col_stride = _matrix_row_stride;
  const auto matrix_tile_row_stride = _tiles_N * matrix_tile_col_stride;

  const TOut* const bptr = (_biases == nullptr) ? nullptr : _biases + start_channel;

  // Loop over batches
  for (int batch = 0; batch < _n_batches; batch++)
  {
    const TIn* const matrix_batch = _matrix_base + start_channel + batch * _matrix_batch_stride;
    TOut* const outptr_batch = _outptr + start_channel + batch * _out_batch_stride;

    for (int tile_i = 0; tile_i < _tiles_M; tile_i++)
    {
      // Compute properties of the row of output tiles
      const int row_pad_bottom = std::max(0, (tile_i + 1)*output_tile_rows - _n_rows);
      const TIn* const matrix_tile_row = matrix_batch + tile_i * matrix_tile_row_stride;
      TOut* const outptr_row = outptr_batch + tile_i * output_tile_rows * _out_row_stride;

      for (int tile_j = 0; tile_j < _tiles_N; tile_j++)
      {
        // Compute property of this specific tile
        const int tile_pad_right = std::max(0, (tile_j + 1)*output_tile_cols - _n_cols);
        const TIn* const matrix_tile = matrix_tile_row + tile_j * matrix_tile_col_stride;
        TOut* const outptr_tile = outptr_row + tile_j * output_tile_cols * _out_col_stride;

        // Perform the transformation
        if (row_pad_bottom || tile_pad_right)
        {
          transform_cropped_tile(
            threadid, n_channels, outptr_tile, matrix_tile, bptr,
            row_pad_bottom, tile_pad_right
          );
        }
        else
        {
          transform_uncropped_tile(
            threadid, n_channels, outptr_tile, matrix_tile, bptr
          );
        }
      }
    }
  }
}

MEMBERFN(void)::transform_uncropped_tile(
  const unsigned int /* threadid unused */,
  const int n_channels,
  TOut * const outptr,
  const TIn * const inptr,
  const TOut * const biases
)
{
  transform_tile(
    n_channels, inptr, _matrix_stride, biases,
    outptr, _out_row_stride, _out_col_stride
  );
}

MEMBERFN(void)::transform_cropped_tile(
  const unsigned int threadid,
  const int n_channels,
  TOut * const outptr,
  const TIn * const inptr,
  const TOut * const biases,
  const int pad_bottom,
  const int pad_right
)
{
  // Transform into working space and then copy the relevant section out.
  TOut *wsptr = static_cast<TOut *>(get_working_space(threadid));
  transform_tile(
    n_channels, inptr, _matrix_stride, biases,
    wsptr, _working_space_row_stride, _working_space_col_stride
  );

  padding::crop_and_copy_tile(
    output_tile_rows, output_tile_cols, n_channels,
    wsptr, _working_space_row_stride, _working_space_col_stride,
    outptr, _out_row_stride, _out_col_stride,
    0u, 0u, pad_bottom, pad_right
  );
}

MEMBERFN(void *)::get_working_space(const unsigned int threadid) const
{
  return _working_space + output_tile_rows * _working_space_row_stride * threadid;
}

}  // namespace winograd
