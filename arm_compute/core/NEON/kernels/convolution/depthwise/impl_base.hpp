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

/*
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 *          NOTE: Header to be included by implementation files only.
 *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

#include <algorithm>
#include "arm_compute/core/NEON/kernels/convolution/depthwise/depthwise.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/utils.hpp"

#pragma once

namespace depthwise
{

template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
int DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::get_output_size(
  const int dim_size, const bool same_padding
)
{
  return iceildiv(dim_size - (same_padding ? 0 : (KC - 1)), SR);
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::DepthwiseConvolution(
  const int n_batches, const int n_input_rows, const int n_input_cols,
  const int n_channels, const bool padding_same,
  const TIn* const weights,
  const TIn* const input,
  TOut* const output
) : _weights(weights), _input(input), _output(output),
    _n_batches(n_batches),
    _n_input_rows(n_input_rows),
    _n_input_cols(n_input_cols),
    _n_channels(n_channels),
    _n_output_rows(get_output_size(n_input_rows, padding_same)),
    _n_output_cols(get_output_size(n_input_cols, padding_same)),
    _n_tile_rows(iceildiv(_n_output_rows, output_tile_rows)),
    _n_tile_cols(iceildiv(_n_output_cols, output_tile_cols)),
    _padding_same(padding_same)
{
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
unsigned int DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::get_window() const
{
  // TODO Later support parallelisation over tile rows.
  return 1;  // _n_tile_rows;
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
void DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::run(
  const unsigned int start,
  const unsigned int stop
)
{
  // TODO Later support parallelisation over tile rows.
  (void) start;
  (void) stop;

  // Compute input striding
  const int input_col_stride = _n_channels;
  const int input_row_stride = _n_input_cols * input_col_stride;
  const int input_batch_stride = _n_input_rows * input_row_stride;

  // Compute output striding
  const int output_col_stride = _n_channels;
  const int output_row_stride = _n_output_cols * output_col_stride;
  const int output_batch_stride = _n_output_rows * output_row_stride;

  // Compute top and bottom padding for input and output
  const int input_pad_top = _padding_same ?
                            ((_n_output_rows - 1)*stride_rows + kernel_rows - _n_input_rows) / 2 : 0;
  const int input_pad_left = _padding_same ?
                             ((_n_output_cols - 1)*stride_cols + kernel_cols - _n_input_cols) / 2 : 0;
  constexpr int tile_overlap = kernel_rows - 1;

  // Perform the convolution by calling `process_tile_row` for each tile row in
  // each batch.
  for (int batch = 0; batch < _n_batches; batch++)
  {
    const TIn* const inptr_batch = _input + batch*input_batch_stride;
    TOut* const outptr_batch = _output + batch*output_batch_stride;

    // Loop over rows of tiles
    for (int tile_i = 0; tile_i < _n_tile_rows; tile_i++)
    {
      // Pointer to the row
      const int input_row_offset = (tile_i == 0) ? 0 : input_pad_top;
      const TIn* const inptr_row = (inptr_batch + ((inner_tile_rows - tile_overlap)*tile_i - input_row_offset)*input_row_stride);
      TOut* const outptr_row = outptr_batch + output_tile_rows * tile_i * output_row_stride;

      // Input padding (top + bottom) for the row
      const int input_row_top = tile_i*(inner_tile_rows - tile_overlap) - input_pad_top;
      const int input_row_bottom = input_row_top + inner_tile_rows;
      const int input_row_pad_top = (tile_i == 0) ? input_pad_top : 0;
      const int input_row_pad_bottom = std::max(0, input_row_bottom - _n_input_rows);

      // Output padding (bottom) for the row
      const int output_row_bottom = (tile_i + 1)*output_tile_rows;
      const int output_row_pad_bottom = std::max(0, output_row_bottom - _n_output_rows);

      // Process the row
      process_tile_row(
        _n_channels, _weights,
        inptr_row, input_row_stride, input_col_stride,
        outptr_row, output_row_stride, output_col_stride,
        input_row_pad_top, input_pad_left, input_row_pad_bottom,
        output_row_pad_bottom,
        _n_tile_cols, _n_input_cols, _n_output_cols
      );
    }
  }
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
void DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::process_tile_row(
  const int n_channels,
  const TIn* const weights,
  const TIn* const inptr,
  const int in_row_stride,
  const int in_col_stride,
  TOut* const outptr,
  const int out_row_stride,
  const int out_col_stride,
  const int row_pad_in_top,
  const int row_pad_in_left,
  const int row_pad_in_bottom,
  const int row_pad_out_bottom,
  const int n_tiles,
  const int n_input_cols,
  const int n_output_cols
)
{
  constexpr int tile_overlap = kernel_cols - 1;

  // Loop over columns of tiles
  for (int tile_j = 0; tile_j < n_tiles; tile_j++)
  {
    // Input padding (left + right) for the tile
    const int t_pad_in_left = (tile_j == 0) ? row_pad_in_left : 0;
    const int t_in_start = tile_j*(inner_tile_cols - tile_overlap) - row_pad_in_left;
    const int t_in_end = t_in_start + inner_tile_cols;
    const int t_pad_in_right = std::max(0, t_in_end - n_input_cols);

    // Output padding (right) for the tile
    const int t_out_end = (tile_j + 1) * output_tile_cols;
    const int t_pad_out_right = std::max(0, t_out_end - n_output_cols);

    // Get pointers into the inputs and outputs
    const int col_offset = (tile_j == 0) ? 0 : row_pad_in_left;
    const TIn* const inptr_col = (inptr + ((inner_tile_cols - tile_overlap)*tile_j - col_offset)*in_col_stride);
    TOut* const outptr_col = outptr + tile_j * output_tile_cols * out_col_stride;

    // Apply the specific tile processing function
    tile_fns[row_pad_in_top][t_pad_in_left][row_pad_in_bottom][t_pad_in_right][row_pad_out_bottom][t_pad_out_right](
      n_channels, weights,
      inptr_col, in_row_stride, in_col_stride,
      outptr_col, out_row_stride, out_col_stride
    );
  }
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
template <
  int in_pad_top, int in_pad_left, int in_pad_bottom, int in_pad_right,
  int out_pad_bottom, int out_pad_right
>
void DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::process_tile(
  const int n_channels,
  const TIn* const weights,
  const TIn* const inptr,
  const int in_row_stride,
  const int in_col_stride,
  TOut* const outptr,
  const int out_row_stride,
  const int out_col_stride
)
{
  // Compute valid ranges of the tile
  constexpr int in_cells_i = inner_tile_rows - in_pad_bottom;
  constexpr int in_cells_j = inner_tile_cols - in_pad_right;
  constexpr int out_cells_i = output_tile_rows - out_pad_bottom;
  constexpr int out_cells_j = output_tile_cols - out_pad_right;

  // Instantiate pointers
  const TIn* inptr_base = inptr;
  const TIn* wptr_base = weights;
  TOut* outptr_base = outptr;

  const int weight_col_stride = n_channels;
  const int weight_row_stride = kernel_cols * n_channels;

  // Perform the depthwise convolution
  int channels_remaining = n_channels;
  for (; channels_remaining; channels_remaining--)
  {
    // Load input tile
    TIn u[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      const TIn* const inptr_row = inptr_base + (i - in_pad_top)*in_row_stride;
      for (int j = 0; j < inner_tile_cols; j++)
      {
        if (i < in_pad_top || in_cells_i <= i ||
            j < in_pad_left || in_cells_j <= j)
        {
          u[i][j] = static_cast<TIn>(0);
        }
        else
        {
          u[i][j] = *(inptr_row + (j - in_pad_left)*in_col_stride);
        }
      }
    }
    inptr_base++;

    // Load weights tile
    TIn w[kernel_rows][kernel_cols];
    for (int i = 0; i < kernel_rows; i++)
    {
      const TIn* const wptr_row = wptr_base + i*weight_row_stride;
      for (int j = 0; j < kernel_cols; j++)
      {
        w[i][j] = *(wptr_row + j*weight_col_stride);
      }
    }
    wptr_base++;

    // Perform the convolution
    TOut v[out_cells_i][out_cells_j];
    for (int out_i = 0; out_i < out_cells_i; out_i++)
    {
      for (int out_j = 0; out_j < out_cells_j; out_j++)
      {
        // Clear the accumulator
        v[out_i][out_j] = static_cast<TOut>(0);

        // Base co-ordinate
        const int base_i = out_i * stride_rows;
        const int base_j = out_j * stride_cols;

        // Fill the accumulator
        for (int in_i = 0; in_i < kernel_rows; in_i++)
        {
          const int i = base_i + in_i;
          for (int in_j = 0; in_j < kernel_cols; in_j++)
          {
            const int j = base_j + in_j;
            v[out_i][out_j] += w[in_i][in_j] * u[i][j];
          }
        }
      }
    }

    // Store the output tile
    for (int i = 0; i < out_cells_i; i++)
    {
      TOut* const outptr_row = outptr_base + i*out_row_stride;
      for (int j = 0; j < out_cells_j; j++)
      {
        *(outptr_row + j*out_col_stride) = v[i][j];
      }
    }
    outptr_base++;
  }
}


// New templated struct used solely as a way to provide tile processing
// specialisations.
template <int OutputTileRows, int OutputTileCols,
          int KernelRows, int KernelCols,
          int StrideRows, int StrideCols,
          typename TIn, typename TOut>
struct DepthwiseConvolutionImpl : public DepthwiseConvolution<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols, TIn, TOut
>
{
  template <
    int in_pad_top, int in_pad_left, int in_pad_bottom, int in_pad_right,
    int out_pad_bottom, int out_pad_right
  >
  static void process_tile(
    const int n_channels,
    const TIn* const weights,
    const TIn* const inptr,
    const int in_row_stride,
    const int in_col_stride,
    TOut* const outptr,
    const int out_row_stride,
    const int out_col_stride
  )
  {
    // By default, redirect to parent. Specialised implementations can be added
    // by overriding this method.
    DepthwiseConvolution<OutputTileRows, OutputTileCols,
                         KernelRows, KernelCols,
                         StrideRows, StrideCols,
                         TIn, TOut>::
      template process_tile<in_pad_top, in_pad_left, in_pad_bottom, in_pad_right,
                            out_pad_bottom, out_pad_right>(
        n_channels,
        weights,
        inptr,
        in_row_stride,
        in_col_stride,
        outptr,
        out_row_stride,
        out_col_stride
    );
  }
};

}  // namespace depthwise
