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
#include "../tensor.hpp"

namespace winograd {
  /* Transform an input tensor into the Winograd domain.
   */
  template <typename T>
  struct Winograd2x2_3x3GemmInput {
    static void execute(
        const T *inptr,
        const Tensor4DShape& input_shape,
        const PaddingType padding_type,
        const int tile_M,
        const int tile_N,
        T *outptr_base,
        const int matrix_stride,
        const int matrix_batch_stride,
        const int matrix_row_stride
    );

    static size_t bytes_read(const Tensor4DShape &input_shape,
                           const Tensor4DShape &output_shape) {
      const int tile_rows = iceildiv(output_shape.n_rows, 2);
      const int tile_cols = iceildiv(output_shape.n_cols, 2);
      return input_shape.n_batches * tile_rows * (16 + 8*(tile_cols - 1)) * input_shape.n_channels * sizeof(T);
    }

    static int flops_performed(const Tensor4DShape &input_shape,
                                const Tensor4DShape &output_shape) {
      const int tile_rows = iceildiv(output_shape.n_rows, 2);
      const int tile_cols = iceildiv(output_shape.n_cols, 2);
      return input_shape.n_batches * tile_rows * (32 + 24*(tile_cols - 1)) * input_shape.n_channels;
    }

    static size_t bytes_written(const Tensor4DShape &input_shape,
                              const Tensor4DShape &output_shape) {
      const int tile_rows = iceildiv(output_shape.n_rows, 2);
      const int tile_cols = iceildiv(output_shape.n_cols, 2);
      const int M = input_shape.n_batches * tile_rows * tile_cols;
      return 16 * M * input_shape.n_channels * sizeof(T);
    }

    protected:
    template <const PaddingType padding, const int pad_bottom, const int pad_right>
    static void process_tile_tensor(
        const int tile_M,      // Number of rows of tiles
        const int tile_N,      // Number of columns of tiles
        int n_channels,  // Number of input channels
        const T* const input,  // Base input pointer (appropriate to batch and channel)
        const int input_row_stride,  // Stride between rows of the input
        const int input_col_stride,  // Stride between columns of the input
        T* const matrix,              // 1st output matrix (appropriate to batch and channel)
        const int matrix_stride,      // Stride between matrices
        const int matrix_row_stride   // Stride between rows of the output matrix
    );

    template <const int pad_top, const int pad_left,
              const int pad_bottom, const int pad_right,
              const int proc_channels>
    static void process_tile_row(
        const int tile_N,      // Number of tiles in the row
        const T* const input,  // Base input pointer (appropriate to batch, channel and row)
        const int input_row_stride,  // Stride between rows of the input
        const int input_col_stride,  // Stride between columns of the input
        T* const matrix,              // 1st output matrix (appropriate to batch, channel and row)
        const int matrix_stride,      // Stride between matrices
        const int matrix_row_stride   // Stride between rows of the output matrix
    );
  };

  template <typename T>
  struct Winograd2x2_3x3GemmInputChannelwise {
    static void execute(
        const T *inptr,
        const Tensor4DShape& input_shape,
        const PaddingType padding_type,
        const int tile_M,
        const int tile_N,
        T *outptr_base,
        const int matrix_stride,
        const int matrix_batch_stride,
        const int matrix_row_stride
    );

    static size_t bytes_read(const Tensor4DShape &input_shape,
                           const Tensor4DShape &output_shape) {
      // We read as many bytes as we write
      return bytes_written(input_shape, output_shape);
    }

    static int flops_performed(const Tensor4DShape &input_shape,
                                const Tensor4DShape &output_shape) {
      const int tile_rows = iceildiv(output_shape.n_rows, 2);
      const int tile_cols = iceildiv(output_shape.n_cols, 2);
      return input_shape.n_batches * tile_rows * 32 * tile_cols * input_shape.n_channels;
    }

    static size_t bytes_written(const Tensor4DShape &input_shape,
                              const Tensor4DShape &output_shape) {
      return winograd::Winograd2x2_3x3GemmInput<T>::bytes_written(input_shape, output_shape);
    }

    protected:
    typedef void (*tilefunc)(int, const T*, int, int, T*, int);
    template <const int pad_top,
              const int pad_left,
              const int pad_bottom,
              const int pad_right>
    static void process_tile(
        int n_channels,  // Number of channels in the tile
        const T* const input_base,
        const int input_row_stride,
        const int input_col_stride,
        T* const matrix_base,
        const int matrix_stride
    );

    private:
    template <const int pad_top,
              const int pad_left,
              const int pad_bottom,
              const int pad_right,
              const int proc_channels>
    static void _process_tile(
        int &n_channels, const T* &inptr,
        const int input_row_stride, const int input_col_stride,
        T* &outptr, const int matrix_stride
    );
  };
}

/*****************************************************************************/
// Include specialised implementations here
#include "input_2x2_3x3/a64_float.hpp"
#include "input_2x2_3x3/a64_float_channelwise.hpp"
/*****************************************************************************/

/*****************************************************************************/
template <typename T>
void winograd::Winograd2x2_3x3GemmInput<T>::execute(
    const T *inptr_base,
    const Tensor4DShape& input_shape,
    const PaddingType padding_type,
    const int tile_M,
    const int tile_N,
    T *outptr_base,
    const int matrix_stride,
    const int matrix_batch_stride,
    const int matrix_row_stride
) {
  // Select an appropriate matrix processing method for the shape and padding
  // of the input tensor.
  typedef void (*tensorfunc)(int, int, int, const T*, int, int, T*, int, int);
  const auto process_tensor = [&padding_type, &input_shape] () -> tensorfunc {
    if (padding_type == PADDING_VALID) {
      const int pad_bottom = input_shape.n_rows % 2;
      const int pad_right = input_shape.n_cols % 2;

      if (pad_bottom == 0 && pad_right == 0) {
        return process_tile_tensor<PADDING_VALID, 0, 0>;
      } else if (pad_bottom == 0 && pad_right == 1) {
        return process_tile_tensor<PADDING_VALID, 0, 1>;
      } else if (pad_bottom == 1 && pad_right == 0) {
        return process_tile_tensor<PADDING_VALID, 1, 0>;
      } else if (pad_bottom == 1 && pad_right == 1) {
        return process_tile_tensor<PADDING_VALID, 1, 1>;
      }
    } else {  // PADDING_SAME
      const int pad_bottom = 1 + input_shape.n_rows % 2;
      const int pad_right = 1 + input_shape.n_cols % 2;

      if (pad_bottom == 1 && pad_right == 1) {
        return process_tile_tensor<PADDING_SAME, 1, 1>;
      } else if (pad_bottom == 1 && pad_right == 2) {
        return process_tile_tensor<PADDING_SAME, 1, 2>;
      } else if (pad_bottom == 2 && pad_right == 1) {
        return process_tile_tensor<PADDING_SAME, 2, 1>;
      } else if (pad_bottom == 2 && pad_right == 2) {
        return process_tile_tensor<PADDING_SAME, 2, 2>;
      }
    }

    printf("%s::%u Uncovered case.\n", __FILE__, __LINE__);
    exit(-1);
    return NULL;  // No function found
  } ();

  // Compute strides
  const int input_row_stride = input_shape.n_cols * input_shape.n_channels;
  const int input_col_stride = input_shape.n_channels;

  // Process each batch of the tensor in turn.
  for (int batch = 0; batch < input_shape.n_batches; batch++) {
    // Work out pointers
    const T *inptr = inptr_base + (batch * input_shape.n_rows *
                                   input_shape.n_cols * input_shape.n_channels);
    T *outptr = outptr_base + batch * matrix_batch_stride;

    // Delegate doing the actual work
    process_tensor(
      tile_M, tile_N, input_shape.n_channels,
      inptr, input_row_stride, input_col_stride,
      outptr, matrix_stride, matrix_row_stride
    );
  }
}

/*****************************************************************************/
template <typename T>
template <const PaddingType padding, const int pad_bottom, const int pad_right>
void winograd::Winograd2x2_3x3GemmInput<T>::process_tile_tensor(
    const int tile_M,      // Number of rows of tiles
    const int tile_N,      // Number of columns of tiles
    int n_channels,  // Number of input channels
    const T* const input,  // Base input pointer (appropriate to batch and channel)
    const int input_row_stride,  // Stride between rows of the input
    const int input_col_stride,  // Stride between columns of the input
    T* const matrix,              // 1st output matrix (appropriate to batch and channel)
    const int matrix_stride,      // Stride between matrices
    const int matrix_row_stride   // Stride between rows of the output matrix
) {
  // Base row processing functions
  typedef void (*rowfunc)(int, const T*, int, int, T*, int, int);
  const rowfunc process_top_row[3] = {
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, 0, pad_right, 1>
      : process_tile_row<1, 1, 0, pad_right, 1>,
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, 0, pad_right, 2>
      : process_tile_row<1, 1, 0, pad_right, 2>,
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, 0, pad_right, 4>
      : process_tile_row<1, 1, 0, pad_right, 4>,
  };
  const rowfunc process_middle_row[3] = {
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, 0, pad_right, 1>
      : process_tile_row<0, 1, 0, pad_right, 1>,
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, 0, pad_right, 2>
      : process_tile_row<0, 1, 0, pad_right, 2>,
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, 0, pad_right, 4>
      : process_tile_row<0, 1, 0, pad_right, 4>,
  };
  const rowfunc process_bottom_row[3] = {
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, pad_bottom, pad_right, 1>
      : process_tile_row<0, 1, pad_bottom, pad_right, 1>,
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, pad_bottom, pad_right, 2>
      : process_tile_row<0, 1, pad_bottom, pad_right, 2>,
    (padding == PADDING_VALID)
      ? process_tile_row<0, 0, pad_bottom, pad_right, 4>
      : process_tile_row<0, 1, pad_bottom, pad_right, 4>,
  };

  // Method to get an input pointer for the given tile row
  const auto get_inptr = [&input, &input_row_stride] (const int tile_i) {
    if (padding == PADDING_VALID) {
      return input + 2 * tile_i * input_row_stride;
    } else {
      return input + (2 * tile_i - (tile_i ? 1 : 0)) * input_row_stride;
    }
  };

  // Wrapper to process a row of tiles, covering all channels.
  const auto process_row =
    [tile_N, input_row_stride, input_col_stride, matrix_stride, matrix_row_stride, n_channels]
    (const rowfunc f[3], const T *inptr, T *outptr) {
      int rem_channels = n_channels;

      // While there remain channels to process continue to process the
      // row.
      for (; rem_channels >= 4; rem_channels -= 4, inptr += 4, outptr += 4) {
        f[2](tile_N, inptr, input_row_stride, input_col_stride, outptr, matrix_stride, matrix_row_stride);
      }
      for (; rem_channels >= 2; rem_channels -= 2, inptr += 2, outptr += 2) {
        f[1](tile_N, inptr, input_row_stride, input_col_stride, outptr, matrix_stride, matrix_row_stride);
      }
      if (rem_channels) {
        f[0](tile_N, inptr, input_row_stride, input_col_stride, outptr, matrix_stride, matrix_row_stride);
      }
  };

  // Process all rows of tiles in the tensor
  for (int tile_i = 0; tile_i < tile_M; tile_i++) {
    T* const m_row = matrix + tile_i * tile_N * matrix_row_stride;
    const T *row_inptr = get_inptr(tile_i);

    if (tile_i == 0) {
      // Top row of the input
      process_row(process_top_row, row_inptr, m_row);
    } else if (tile_i == tile_M - 1) {
      // Bottom row of the input
      process_row(process_bottom_row, row_inptr, m_row);
    } else {
      // Any other row of the input
      process_row(process_middle_row, row_inptr, m_row);
    }
  }
}

/*****************************************************************************/
template <typename T>
template <const int pad_top, const int pad_left,
          const int pad_bottom, const int pad_right,
          const int proc_channels>
void winograd::Winograd2x2_3x3GemmInput<T>::process_tile_row(
    const int tile_N,      // Number of tiles in the row
    const T* const input,  // Base input pointer (appropriate to batch, channel and row)
    const int input_row_stride,  // Stride between rows of the input
    const int input_col_stride,  // Stride between columns of the input
    T* const matrix,              // 1st output matrix (appropriate to batch, channel and row)
    const int matrix_stride,      // Stride between matrices
    const int matrix_row_stride   // Stride between rows of the output matrix
) {
  // Construct copies of the pointers
  const T *inptr = input;
  T *outptr = matrix;

  // Storage for the tensors x, X.T x, and X.T x X.
  T x[4][4][proc_channels], XTx[4][4][proc_channels], XTxX[4][4][proc_channels];

  // For every tile in the row
  for (int tile_j = 0; tile_j < tile_N; tile_j++) {
    // Determine the padding for the tile
    const int tile_pad_left = (tile_j == 0) ? pad_left : 0;
    const int tile_pad_right = (tile_j == tile_N - 1) ? pad_right : 0;

    // Load tile values. If this is the first tile in the row then we must load
    // all values, otherwise we can just load the final two columns of the input.
    for (int i = 0; i < 4; i++) {
      for (int j = ((tile_j == 0) ? 0 : 2); j < 4; j++) {
        // Fill with padding if required
        if (i < pad_top || 4 - pad_bottom <= i ||
            j < tile_pad_left || 4 - tile_pad_right <= j) {
          for (int c = 0; c < proc_channels; c++) {
            x[i][j][c] = static_cast<T>(0);  // Padding
          }
        } else {
          // Load values, note that the initial padding offsets the pointer we
          // were provided.
          for (int c = 0; c < proc_channels; c++) {
            const int row_offset = (i - pad_top) * input_row_stride;
            const int col_offset = (j - tile_pad_left) * input_col_stride;
            x[i][j][c] = inptr[row_offset + col_offset + c];
          }
        }
      }
    }

    // Compute the matrix X.T x.  Note, can elide operations depending on the
    // padding. Furthermore, if this isn't the left-most tile we can skip half
    // of the operations by copying results from the previous version of X.T x.
    // This latter optimisation can be simplified by unrolling the outermost
    // loop by two and by renaming the registers containing XTx.
    if (tile_j == 0) {
      for (int j = 0; j < 4; j++) {
        for (int c = 0; c < proc_channels; c++) {
          XTx[0][j][c] =  x[0][j][c] - x[2][j][c];
          XTx[1][j][c] =  x[1][j][c] + x[2][j][c];
          XTx[2][j][c] = -x[1][j][c] + x[2][j][c];
          XTx[3][j][c] =  x[1][j][c] - x[3][j][c];
        }
      }
    } else {
      for (int j = 0; j < 2; j++) {
        for (int c = 0; c < proc_channels; c++) {
          XTx[0][j][c] = XTx[0][j + 2][c];
          XTx[1][j][c] = XTx[1][j + 2][c];
          XTx[2][j][c] = XTx[2][j + 2][c];
          XTx[3][j][c] = XTx[3][j + 2][c];
        }
      }
      for (int j = 2; j < 4; j++) {
        for (int c = 0; c < proc_channels; c++) {
          XTx[0][j][c] =  x[0][j][c] - x[2][j][c];
          XTx[1][j][c] =  x[1][j][c] + x[2][j][c];
          XTx[2][j][c] = -x[1][j][c] + x[2][j][c];
          XTx[3][j][c] =  x[1][j][c] - x[3][j][c];
        }
      }
    }

    // Compute the matrix X.T x X. Note, can elide operations based on the
    // padding.
    for (int i = 0; i < 4; i++) {
      for (int c = 0; c < proc_channels; c++) {
        XTxX[i][0][c] =  XTx[i][0][c] - XTx[i][2][c];
        XTxX[i][1][c] =  XTx[i][1][c] + XTx[i][2][c];
        XTxX[i][2][c] = -XTx[i][1][c] + XTx[i][2][c];
        XTxX[i][3][c] =  XTx[i][1][c] - XTx[i][3][c];
      }
    }

    // Store the output matrix (X.T x X)
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        // Get a pointer to the relevant output matrix
        T *mptr = outptr + (i*4 + j)*matrix_stride;

        // Write out the channels
        for (int c = 0; c < proc_channels; c++) {
          mptr[c] = XTxX[i][j][c];
        }
      }
    }

    // Update the pointers
    inptr += input_col_stride * ((tile_j == 0 && pad_left) ? 1 : 2);
    outptr += matrix_row_stride;
  }
}

/*****************************************************************************/
template <typename T>
void winograd::Winograd2x2_3x3GemmInputChannelwise<T>::execute(
    const T *inptr,
    const Tensor4DShape& input_shape,
    const PaddingType padding_type,
    const int tile_M,
    const int tile_N,
    T *outptr_base,
    const int matrix_stride,
    const int matrix_batch_stride,
    const int matrix_row_stride
) {
  const int n_channels = input_shape.n_channels;
  const int input_col_stride = n_channels;
  const int input_row_stride = input_shape.n_cols * input_col_stride;

  // Determine the padding and hence select appropriate methods for each tile.
  tilefunc fs[3][3];

  if (padding_type == PADDING_VALID) {
    constexpr int pad_top = 0;
    constexpr int pad_left = 0;
    const int pad_right = input_shape.n_cols % 2 == 0;

    fs[0][0] = process_tile<pad_top, pad_left, 0, 0>;
    fs[0][1] = process_tile<pad_top, 0, 0, 0>;
    fs[0][2] = (pad_right) ? process_tile<pad_top, 0, 0, 0> : process_tile<pad_top, 0, 0, 1>;

    fs[1][0] = process_tile<0, pad_left, 0, 0>;
    fs[1][1] = process_tile<0, 0, 0, 0>;
    fs[1][2] = (pad_right) ? process_tile<0, 0, 0, 0> : process_tile<0, 0, 0, 1>;

    if (input_shape.n_rows % 2 == 0) {
      constexpr int pad_bottom = 0;
      fs[2][0] = process_tile<0, pad_left, pad_bottom, 0>;
      fs[2][1] = process_tile<0, 0, pad_bottom, 0>;
      fs[2][2] = (pad_right) ? process_tile<0, 0, pad_bottom, 0> : process_tile<0, 0, pad_bottom, 1>;
    } else {
      constexpr int pad_bottom = 1;
      fs[2][0] = process_tile<0, pad_left, pad_bottom, 0>;
      fs[2][1] = process_tile<0, 0, pad_bottom, 0>;
      fs[2][2] = (pad_right) ? process_tile<0, 0, pad_bottom, 0> : process_tile<0, 0, pad_bottom, 1>;
    }
  } else {
    constexpr int pad_top = 1;
    constexpr int pad_left = 1;
    const int pad_right = input_shape.n_cols % 2 == 0;

    fs[0][0] = process_tile<pad_top, pad_left, 0, 0>;
    fs[0][1] = process_tile<pad_top, 0, 0, 0>;
    fs[0][2] = (pad_right) ? process_tile<pad_top, 0, 0, 1> : process_tile<pad_top, 0, 0, 2>;

    fs[1][0] = process_tile<0, pad_left, 0, 0>;
    fs[1][1] = process_tile<0, 0, 0, 0>;
    fs[1][2] = (pad_right) ? process_tile<0, 0, 0, 1> : process_tile<0, 0, 0, 2>;

    if (input_shape.n_rows % 2 == 0) {
      constexpr int pad_bottom = 1;
      fs[2][0] = process_tile<0, pad_left, pad_bottom, 0>;
      fs[2][1] = process_tile<0, 0, pad_bottom, 0>;
      fs[2][2] = (pad_right) ? process_tile<0, 0, pad_bottom, 1> : process_tile<0, 0, pad_bottom, 2>;
    } else {
      constexpr int pad_bottom = 2;
      fs[2][0] = process_tile<0, pad_left, pad_bottom, 0>;
      fs[2][1] = process_tile<0, 0, pad_bottom, 0>;
      fs[2][2] = (pad_right) ? process_tile<0, 0, pad_bottom, 1> : process_tile<0, 0, pad_bottom, 2>;
    }
  }

  // Process each tile in turn
  for (int batch = 0; batch < input_shape.n_batches; batch++) {
    const T* const input_base_batch = inptr + batch*input_shape.n_rows*input_shape.n_cols*n_channels;

    for (int tile_i = 0; tile_i < tile_M; tile_i++) {
      const int row_offset = (tile_i == 0) ? 0 : ((padding_type == PADDING_VALID) ? 0 : 1);
      const T* const input_base_row = input_base_batch + (2*tile_i - row_offset)*input_shape.n_cols*n_channels;

      // Select the set of functions for the row
      const int fs_i = (tile_i == 0) ? 0 : ((tile_i < tile_M - 1) ? 1 : 2);

      for (int tile_j = 0; tile_j < tile_N; tile_j++) {
        // Select the function for the column
        const int fs_j = (tile_j == 0) ? 0 : ((tile_j < tile_N - 1) ? 1 : 2);
        const auto f = fs[fs_i][fs_j];

        // Get pointers into the input and outputs
        const int col_offset = (tile_j == 0) ? 0 : ((padding_type == PADDING_VALID) ? 0 : 1);
        const T* const input_base_col = input_base_row + (2*tile_j - col_offset)*n_channels;
        T* const matrix_base = outptr_base + batch*matrix_batch_stride + (tile_i*tile_N + tile_j)*matrix_row_stride;
        f(n_channels, input_base_col, input_row_stride, input_col_stride,
          matrix_base, matrix_stride);
      }
    }
  }
}

template <typename T>
template <const int pad_top,
          const int pad_left,
          const int pad_bottom,
          const int pad_right>
void winograd::Winograd2x2_3x3GemmInputChannelwise<T>::process_tile(
    int n_channels,  // Number of channels in the tile
    const T* const input_base,
    const int input_row_stride,
    const int input_col_stride,
    T* const matrix_base,
    const int matrix_stride
) {
  // Copy pointers
  const T *inptr = input_base;
  T *outptr = matrix_base;

  // Process channels (modifies inptr, outptr and n_channels)
  _process_tile<pad_top, pad_left, pad_bottom, pad_right, 4>(
    n_channels, inptr, input_row_stride, input_col_stride,
    outptr, matrix_stride
  );
  _process_tile<pad_top, pad_left, pad_bottom, pad_right, 2>(
    n_channels, inptr, input_row_stride, input_col_stride,
    outptr, matrix_stride
  );
  _process_tile<pad_top, pad_left, pad_bottom, pad_right, 1>(
    n_channels, inptr, input_row_stride, input_col_stride,
    outptr, matrix_stride
  );
}

template <typename T>
template <const int pad_top,
          const int pad_left,
          const int pad_bottom,
          const int pad_right,
          const int proc_channels>
void winograd::Winograd2x2_3x3GemmInputChannelwise<T>::_process_tile(
    int &n_channels,
    const T* &inptr, const int input_row_stride, const int input_col_stride,
    T* &outptr, const int matrix_stride
) {
  // We use 4 pointers to point at matrices 0, 4, 8 and 12 and use three
  // offsets to access the intermediate matrices.
  T* outptrs[4] = {
    outptr,
    outptr + matrix_stride * 4,
    outptr + matrix_stride * 8,
    outptr + matrix_stride * 12
  };

  // The matrix X; zeroed to account for padding.
  T x[4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      x[i][j] = 0;
    }
  }

  // The matrices X.T x and U
  T XTx[4][4], U[4][4];

  // Now progress through each channel
  for (; n_channels >= proc_channels; n_channels -= proc_channels) {
    for (int n = 0; n < proc_channels; n++) {
      // Load the matrix X
      for (int cell_i = pad_top, i = 0; cell_i < 4 - pad_bottom; cell_i++, i++) {
        for (int cell_j = pad_left, j = 0; cell_j < 4 - pad_right; cell_j++, j++) {
          x[cell_i][cell_j] = inptr[i*input_row_stride + j*input_col_stride];
        }
      }
      inptr++;

      // Compute the matrix X.T
      for (int j = 0; j < 4; j++) {
        XTx[0][j] = x[0][j] - x[2][j];
        XTx[1][j] = x[1][j] + x[2][j];
        XTx[2][j] = x[2][j] - x[1][j];
        XTx[3][j] = x[1][j] - x[3][j];
      }

      // Hence compute the matrix U
      for (int i = 0; i < 4; i++) {
        U[i][0] = XTx[i][0] - XTx[i][2];
        U[i][1] = XTx[i][1] + XTx[i][2];
        U[i][2] = XTx[i][2] - XTx[i][1];
        U[i][3] = XTx[i][1] - XTx[i][3];
      }

      // Store the matrix U
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          outptrs[i][j * matrix_stride] = U[i][j];
        }
        outptrs[i]++;
      }
    }
  }

  // Update the output pointer for future calls
  outptr = outptrs[0];
}
