/*
 * Copyright (c) 2018 ARM Limited.
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

namespace winograd
{

namespace
{

template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
class InputTransformImpl
{
  public:
    /** Apply the transform to a tensor. */
    static void execute(
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
    );

  private:
    static void process_tile_row(
      const int tile_N,
      int n_channels,
      const T* const input_base,
      const int input_row_stride,
      const int input_col_stride,
      T* const matrix_base,
      const int matrix_stride,
      const int matrix_row_stride,
      const int row_pad_top,
      const int row_pad_left,
      const int row_pad_bottom,
      const int n_cols
    );

    // Tile overlaps
    static constexpr int overlap_rows = KernelRows - 1;
    static constexpr int overlap_cols = KernelCols - 1;

    // Maximum padding and number of distinct paddings
    static constexpr int max_pad_top = KernelRows / 2;
    static constexpr int n_pad_top = 1 + iceildiv(max_pad_top, InnerTileRows - overlap_rows);

    static constexpr int max_pad_left = KernelCols / 2;
    static constexpr int n_pad_left = 1 + iceildiv(max_pad_left, InnerTileCols - overlap_cols);

    static constexpr int n_pad_bottom = InnerTileRows;
    static constexpr int n_pad_right = InnerTileCols;

    /** Process a single tile of the input tensor. */
    template <int pad_top, int pad_left, int pad_bottom, int pad_right>
    static void process_tile(int, const T*, int, int, T*, int);

    // Array of methods to transform tiles of the input tensor.
    typedef void (*TileFn)(int, const T*, int, int, T*, int);
    static const TileFn
      tile_fns[n_pad_top][n_pad_left][n_pad_bottom][n_pad_right];
};


template <int KernelRows, int InnerTileRows, typename T>
class InputTransformImpl<KernelRows, 1, InnerTileRows, 1, T>
{
  public:
    /** Apply the transform to a tensor. */
    static void execute(
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
    );
};

}  // namespace (anonymous)

template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
class InputTransform
{
  public:
  /***********************************************************************/
  /** Create an InputTransform operator fixed on a given problem and set of
   * pointers.
   */
  InputTransform(
      const T* const input,        /** Input tensor data */
      const int n_batches,         /** Number of batches in input tensor. */
      const int n_rows,            /** Number of rows in input tensor. */
      const int n_cols,            /** Number of columns in input tensor. */
      const int n_channels,        /** Number of channels in input tensor. */
      const PaddingType padding,   /** Padding type. */
      T* const output,             /** Base of output matrices. */
      const int matrix_stride,     /** Stride between output matrices. */
      const int matrix_row_stride, /** Stride within matrices. */
      const int in_batch_stride=0, /** Stride between input batches. */
      const int in_row_stride=0,   /** Stride between input rows. */
      const int in_col_stride=0    /** Stride between input columns. */
  );

  /** Get the window of work a given operator can perform. */
  unsigned int get_window() const;
  static constexpr unsigned int WINDOW_BLOCK = 16;  // Base size of window

  /** Perform work upon a window of the input. */
  void run(const unsigned int start, const unsigned int stop);

  /** Apply the transform to a tensor. */
  static void execute(
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
  );

  protected:
    using Transform = InputTransformImpl<KernelRows, KernelCols, InnerTileRows, InnerTileCols, T>;

    /* Member values for instance-based API. */
    const T* const _inptr;
    T* const _outptr;
    const int _n_batches, _n_rows, _n_cols, _n_channels, _matrix_stride,
              _matrix_row_stride, _tiles_M, _tiles_N;
    const int _in_col_stride, _in_row_stride, _in_batch_stride;
    const PaddingType _padding_type;
};

}  // namespace winograd
