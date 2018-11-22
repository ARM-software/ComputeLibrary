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
class OutputTransformImplTiles
{
  public:
    typedef void (*TileFn)(
      const int n_channels,         /** @param[in] Number of channels in output tensor */
      const T* const matrix_base,   /** @param[in] Base pointer to Winograd output matrices. */
      const int matrix_stride,      /** @param[in] Stride between matrices in the output space. */
      const T* const biases,        /** @param[in] Pointer to bias vector (may be nullptr). */
      T* const output,              /** @param[out] Pointer to output tensor. */
      const int output_row_stride,  /** @param[in] Stride across rows of the output tensor. */
      const int output_col_stride,  /** @param[in] Stride between columns of the output tensor. */
      const int _pad_bottom,        /** @param[in] Bottom padding for unspecialised tiles. */
      const int _pad_right          /** @param[in] Right padding for unspecialised tiles. */
    );

    static TileFn get_tile_specialization(
      const int pad_bottom,
      const int pad_right
    );

    static constexpr unsigned int OutputTileRows = InnerTileRows - KernelRows + 1;
    static constexpr unsigned int OutputTileCols = InnerTileCols - KernelCols + 1;

  private:
    static constexpr unsigned int n_pad_bottom = OutputTileRows - 1;
    static constexpr unsigned int n_pad_right = OutputTileCols - 1;

    static const TileFn tilefn_generic;   /** Generic tile processing function. */
    static const TileFn tilefn_unpadded;  /** Tile processor for unpadded tiles. */
    static const TileFn tilefn_bottom_padded[n_pad_bottom];  /** Bottom padding only. */
    static const TileFn tilefn_right_padded[n_pad_right];    /** Right padding only. */
};

template <int KernelCols, int InnerTileCols, typename T>
class OutputTransformImplTiles<1, KernelCols, 1, InnerTileCols, T>
{
  public:
    typedef void (*TileFn)(
      const int n_channels,         /** @param[in] Number of channels in output tensor */
      const T* const matrix_base,   /** @param[in] Base pointer to Winograd output matrices. */
      const int matrix_stride,      /** @param[in] Stride between matrices in the output space. */
      const T* const biases,        /** @param[in] Pointer to bias vector (may be nullptr). */
      T* const output,              /** @param[out] Pointer to output tensor. */
      const int output_row_stride,  /** @param[in] Stride across rows of the output tensor. */
      const int output_col_stride,  /** @param[in] Stride between columns of the output tensor. */
      const int _pad_bottom,        /** @param[in] Bottom padding for unspecialised tiles. */
      const int _pad_right          /** @param[in] Right padding for unspecialised tiles. */
    );

    static TileFn get_tile_specialization(
      const int pad_bottom,
      const int pad_right
    );

    static constexpr unsigned int OutputTileRows = 1;
    static constexpr unsigned int OutputTileCols = InnerTileCols - KernelCols + 1;

  private:
    static constexpr unsigned int n_pad_right = OutputTileCols - 1;

    static const TileFn tilefn_unpadded;  /** Tile processor for unpadded tiles. */
    static const TileFn tilefn_right_padded[n_pad_right];    /** Right padding only. */
};

template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
class OutputTransformImpl
{
  private:
    static void process_tile_row(
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
    );

    using Tiles = OutputTransformImplTiles<
      KernelRows, KernelCols, InnerTileRows, InnerTileCols, T
    >;

  public:
    /** Apply the output transform to a tensor. */
    static void execute(
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
    );

    static constexpr unsigned int OutputTileRows = Tiles::OutputTileRows;
    static constexpr unsigned int OutputTileCols = Tiles::OutputTileCols;
};

template <int KernelRows, int InnerTileRows, typename T>
class OutputTransformImpl<KernelRows, 1, InnerTileRows, 1, T>
{
  public:
    /** Apply the output transform to a tensor. */
    static void execute(
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
    );

    static constexpr unsigned int OutputTileRows = InnerTileRows - KernelRows + 1;
    static constexpr unsigned int OutputTileCols = 1;
};

}  // namespace (anonymous)

template <int KernelRows, int KernelCols, int InnerTileRows, int InnerTileCols, typename T>
class OutputTransform
{
  public:
    /***********************************************************************/
    /** Create an OutputTransform operator fixed on a given problem and set
     * of pointers.
     */
    OutputTransform(
      const T* const matrix_base,   /** Pointer to base of matrices. */
      const int matrix_stride,      /** Stride between matrices. */
      const int matrix_row_stride,  /** Stride within a matrix. */
      const T* const biases,        /** Pointer to biases vector. */
      T* const output,              /** Pointer to output tensor. */
      const int n_batches,          /** Number of batches in output tensor. */
      const int n_rows,             /** Number of rows in output tensor. */
      const int n_cols,             /** Number of columns in output tensor. */
      const int n_channels,         /** Number of channels in output tensor. */
      const int out_batch_stride=0, /** Output batch stride. */
      const int out_row_stride=0,   /** Output row stride. */
      const int out_col_stride=0    /** Output column stride. */
    );

    /** Get the window of work a given operator can perform. */
    unsigned int get_window() const;
    static constexpr unsigned int WINDOW_BLOCK = 16;  // Base size of window

    /** Perform work upon a window of the input. */
    void run(const unsigned int start, const unsigned int stop);

    /** Apply the transform to create a tensor. */
    static void execute(
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
    );

  private:
    using Transform = OutputTransformImpl<
      KernelRows, KernelCols, InnerTileRows, InnerTileCols, T
    >;

    static constexpr unsigned int OutputTileRows = Transform::OutputTileRows;
    static constexpr unsigned int OutputTileCols = Transform::OutputTileCols;

    /** Member constants for instances of the transform. */
    const T* const _matrix_base;
    const T* const _biases;
    const int _matrix_stride, _matrix_row_stride;
    T* const _outptr;
    const int _n_batches, _n_rows, _n_cols, _n_channels, _tile_M, _tile_N;
    const int _out_col_stride, _out_row_stride, _out_batch_stride;
};

}  // namespace winograd

