/*
 * Copyright (c) 2018-2019 ARM Limited.
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

const unsigned int CHANNEL_BLOCK = 16;

namespace
{
  inline int pad_along_dim(
    const bool padding_same,
    const int kernel_dim,
    const int stride_dim,
    const int input_dim
  )
  {
    if (!padding_same)
      return 0;
    if (input_dim % stride_dim)
      return std::max(kernel_dim - (input_dim % stride_dim), 0);
    else
      return std::max(kernel_dim - stride_dim, 0);
  }
}  // namespace

template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
int DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::get_output_size(
  const int dim_size, const bool same_padding
)
{
  return iceildiv(dim_size - (same_padding ? 0 : (KC - 1)), SR);
}

template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
int DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::get_output_size(
  const int dim_size, const unsigned int padding_before, const unsigned int padding_after
)
{
  return iceildiv(dim_size + padding_before + padding_after - KR + 1, SR);
}

template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::DepthwiseConvolution(
  const int n_batches, const int n_input_rows, const int n_input_cols,
  const int n_channels, const bool padding_same,
  const TIn* const weights,
  const TIn* const input,
  TOut* const output,
  const int weight_col_stride,
  const int weight_row_stride,
  const int input_col_stride,
  const int input_row_stride,
  const int input_batch_stride,
  const int output_col_stride,
  const int output_row_stride,
  const int output_batch_stride
) : DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>(
  n_batches, n_input_rows, n_input_cols,
  n_channels,
  pad_along_dim(padding_same, KR, SR, n_input_rows) / 2,  /* top padding */
  pad_along_dim(padding_same, KC, SC, n_input_cols) / 2,  /* left padding */
  iceildiv(pad_along_dim(padding_same, KR, SR, n_input_rows), 2),  /* bottom padding */
  iceildiv(pad_along_dim(padding_same, KC, SC, n_input_cols), 2),  /* right padding */
  weights, input, output,
  weight_col_stride, weight_row_stride,
  input_col_stride, input_row_stride, input_batch_stride,
  output_col_stride, output_row_stride, output_batch_stride
)
{
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::DepthwiseConvolution(
  const int n_batches, const int n_input_rows, const int n_input_cols,
  const int n_channels,
  const unsigned int padding_top,
  const unsigned int padding_left,
  const unsigned int padding_bottom,
  const unsigned int padding_right,
  const TIn* const weights,
  const TIn* const input,
  TOut* const output,
  const int weight_col_stride,
  const int weight_row_stride,
  const int input_col_stride,
  const int input_row_stride,
  const int input_batch_stride,
  const int output_col_stride,
  const int output_row_stride,
  const int output_batch_stride
) : _weights(weights), _input(input), _output(output),
    _n_batches(n_batches),
    _n_input_rows(n_input_rows),
    _n_input_cols(n_input_cols),
    _n_channels(n_channels),
    _n_output_rows(get_output_size(n_input_rows, padding_top, padding_bottom)),
    _n_output_cols(get_output_size(n_input_cols, padding_left, padding_right)),
    _n_tile_rows(iceildiv(_n_output_rows, output_tile_rows)),
    _n_tile_cols(iceildiv(_n_output_cols, output_tile_cols)),
    _padding_top(padding_top),
    _padding_left(padding_left),
    _padding_bottom(padding_bottom),
    _padding_right(padding_right),
    _weight_col_stride(weight_col_stride ? weight_col_stride : _n_channels),
    _weight_row_stride(weight_row_stride ? weight_row_stride : KC * _weight_col_stride),
    _input_col_stride(input_col_stride ? input_col_stride : _n_channels),
    _input_row_stride(input_row_stride ? input_row_stride : _n_input_cols * _input_col_stride),
    _input_batch_stride(input_batch_stride ? input_batch_stride : _n_input_rows * _input_row_stride),
    _output_col_stride(output_col_stride ? output_col_stride : _n_channels),
    _output_row_stride(output_row_stride ? output_row_stride : _n_output_cols * _output_col_stride),
    _output_batch_stride(output_batch_stride ? output_batch_stride : _n_output_rows * _output_row_stride),
    _input_offset(0), _weights_offset(0)
{
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
unsigned int DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::get_window() const
{
  // Parallelise over blocks of channels.
  return iceildiv(_n_channels, CHANNEL_BLOCK);
}

template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
void DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::set_offsets(int input_offset, int weights_offset)
{
    _input_offset = input_offset;
    _weights_offset = weights_offset;
}

template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
void DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::run(
  const unsigned int start,
  const unsigned int stop
)
{
  // Parallelise over blocks of channels
  const auto start_channel = CHANNEL_BLOCK * start;
  const auto stop_channel = std::min<unsigned int>(_n_channels, CHANNEL_BLOCK * stop);

  // Compute top and bottom padding for input and output
  const int input_pad_top = _padding_top;
  const int input_pad_left = _padding_left;
  constexpr int tile_overlap = kernel_rows - stride_rows;

  // Perform the convolution by calling `process_tile_row` for each tile row in
  // each batch.
  for (int batch = 0; batch < _n_batches; batch++)
  {
    const TIn* const inptr_batch = _input + batch*_input_batch_stride;
    TOut* const outptr_batch = _output + batch*_output_batch_stride;

    // Loop over rows of tiles
    for (int tile_i = 0; tile_i < _n_tile_rows; tile_i++)
    {
      // Pointer to the row
      const int input_row_offset = (tile_i == 0) ? 0 : input_pad_top;
      const TIn* const inptr_row = (inptr_batch + ((inner_tile_rows - tile_overlap)*tile_i - input_row_offset)*_input_row_stride);
      TOut* const outptr_row = outptr_batch + output_tile_rows * tile_i * _output_row_stride;

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
        stop_channel - start_channel,
        _weights + start_channel, _weight_row_stride, _weight_col_stride,
        inptr_row + start_channel, _input_row_stride, _input_col_stride,
        outptr_row + start_channel, _output_row_stride, _output_col_stride,
        input_row_pad_top, input_pad_left, input_row_pad_bottom,
        output_row_pad_bottom,
        _n_tile_cols, _n_input_cols, _n_output_cols,
        _input_offset, _weights_offset
      );
    }
  }
}


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
void DepthwiseConvolution<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::process_tile_row(
  const int n_channels,
  const TIn* const weights,
  const int weight_row_stride,
  const int weight_col_stride,
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
  const int n_output_cols,
  const int input_offset,
  const int weights_offset
)
{
  constexpr int tile_overlap = kernel_cols - stride_cols;

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
    const bool pad_top = row_pad_in_top > 0;
    const bool pad_left = t_pad_in_left > 0;
    const bool pad_bottom = row_pad_in_bottom || row_pad_out_bottom;
    const bool pad_right = t_pad_in_right || t_pad_out_right;

    const TileFn tilefn = [&] () {
      if (!pad_top && !pad_left && !pad_bottom && !pad_right)
      {
        // No padding
        return tilefn_unpadded;
      }
      else if (pad_top && !pad_left && !pad_bottom && !pad_right)
      {
        // Padding on the top only, subtract off the minimum expected padding in
        // order to index into the array of specialised methods.
        const int index = row_pad_in_top - min_in_pad_top;
        return tilefn_top[index];
      }
      else if (!pad_top && pad_left && !pad_bottom && !pad_right)
      {
        // Padding on the left only, subtract off the minimum expected padding in
        // order to index into the array of specialised methods.
        const int index = t_pad_in_left - min_in_pad_left;
        return tilefn_left[index];
      }
      else if (!pad_top && !pad_left && pad_bottom && !pad_right)
      {
        // Padding on the bottom only
        return tilefn_bottom[row_pad_in_bottom][row_pad_out_bottom];
      }
      else if (!pad_top && !pad_left && !pad_bottom && pad_right)
      {
        // Padding on the right only
        return tilefn_right[t_pad_in_right][t_pad_out_right];
      }
      else
      {
        // Otherwise use generic tile processing method.
        return tilefn_generic;
      }
    }();

    tilefn(
      n_channels,
      weights, weight_row_stride, weight_col_stride,
      inptr_col, in_row_stride, in_col_stride,
      outptr_col, out_row_stride, out_col_stride,
      row_pad_in_top, t_pad_in_left, row_pad_in_bottom, t_pad_in_right,
      row_pad_out_bottom, t_pad_out_right, input_offset, weights_offset
    );
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
  typedef DepthwiseConvolution<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    TIn, TOut
  > DWC;

  /** Perform the depthwise convolution of a tile.
   *
   * @param[in] n_channels Number of channels.
   * @param[in] weights Pointer to Height x Width x Channels ordered weights.
   * @param[in] inptr Pointer to the top-left unpadded value of the tile.
   * @param[in] in_row_stride Stride between rows of the input tensor.
   * @param[in] in_col_stride Stride between columns of the input tensor.
   * @param[out] outptr Pointer to the top-left output value for the tile.
   * @param[in] out_row_stride Stride between rows of the output tensor.
   * @param[in] out_col_stride Stride between columns of the output tensor.
   *
   * The following parameters may be ignored if the function has been
   * specialised for specific padding constraints.
   *
   * @param[in] _in_pad_top Padding to apply to top of input tile.
   * @param[in] _in_pad_left Padding to apply to left of input tile.
   * @param[in] _in_pad_bottom Padding to apply to bottom of input tile.
   * @param[in] _in_pad_right Padding to apply to right of input tile.
   * @param[in] _out_pad_bottom Null cells at bottom of output tile.
   * @param[in] _out_pad_right Null cells at right of output tile.
   */
  template <
    bool Specialize=false,  // Specialize (or not) the method
    int InPadTop=0,         // If specialized, top padding
    int InPadLeft=0,        // If specialized, left padding
    int InPadBottom=0,      // If specialized, bottom padding
    int InPadRight=0,       // If specialized, right padding
    int OutPadBottom=0,     // If specialized, bottom output padding
    int OutPadRight=0       // If specialized, bottom right padding
  >
  static void process_tile(
    const int n_channels,
    const TIn* const weights,
    const int weight_row_stride,
    const int weight_col_stride,
    const TIn* const inptr,
    const int in_row_stride,
    const int in_col_stride,
    TOut* const outptr,
    const int out_row_stride,
    const int out_col_stride,
    const int in_pad_top=0,
    const int in_pad_left=0,
    const int in_pad_bottom=0,
    const int in_pad_right=0,
    const int out_pad_bottom=0,
    const int out_pad_right=0,
    const int input_offset=0,
    const int weights_offset=0
  );
};


template <int OTR, int OTC, int KR, int KC, int SR, int SC, typename TIn, typename TOut>
template <
  bool Specialize,
  int InPadTop, int InPadLeft, int InPadBottom, int InPadRight,
  int OutPadBottom, int OutPadRight
>
void DepthwiseConvolutionImpl<OTR, OTC, KR, KC, SR, SC, TIn, TOut>::process_tile(
  const int n_channels,
  const TIn *__restrict__ const weights,
  const int weight_row_stride,
  const int weight_col_stride,
  const TIn *__restrict__ const inptr,
  const int in_row_stride,
  const int in_col_stride,
  TOut *__restrict__ const outptr,
  const int out_row_stride,
  const int out_col_stride,
  const int _in_pad_top,
  const int _in_pad_left,
  const int _in_pad_bottom,
  const int _in_pad_right,
  const int _out_pad_bottom,
  const int _out_pad_right,
  const int _input_offset,
  const int _weights_offset
)
{
  constexpr auto inner_tile_rows = DWC::inner_tile_rows;
  constexpr auto inner_tile_cols = DWC::inner_tile_cols;
  constexpr auto kernel_rows = DWC::kernel_rows;
  constexpr auto kernel_cols = DWC::kernel_cols;
  constexpr auto output_tile_rows = DWC::output_tile_rows;
  constexpr auto output_tile_cols = DWC::output_tile_cols;
  constexpr auto stride_rows = DWC::stride_rows;
  constexpr auto stride_cols = DWC::stride_cols;

  // Extract parameters
  const int in_pad_top = Specialize ? InPadTop : _in_pad_top;
  const int in_pad_left = Specialize ? InPadLeft : _in_pad_left;
  const int in_pad_bottom = Specialize ? InPadBottom : _in_pad_bottom;
  const int in_pad_right = Specialize ? InPadRight : _in_pad_right;
  const int out_pad_bottom = Specialize ? OutPadBottom : _out_pad_bottom;
  const int out_pad_right = Specialize ? OutPadRight : _out_pad_right;

  // Compute valid ranges of the tile
  const int in_cells_i = inner_tile_rows - in_pad_bottom;
  const int in_cells_j = inner_tile_cols - in_pad_right;
  const int out_cells_i = output_tile_rows - out_pad_bottom;
  const int out_cells_j = output_tile_cols - out_pad_right;

  // Instantiate pointers
  const TIn* __restrict__ inptr_base = inptr;
  const TIn* __restrict__ wptr_base = weights;
  TOut* __restrict__ outptr_base = outptr;

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
    TOut v[output_tile_rows][output_tile_cols];
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
      TOut* __restrict__ const outptr_row = outptr_base + i*out_row_stride;
      for (int j = 0; j < out_cells_j; j++)
      {
        *(outptr_row + j*out_col_stride) = v[i][j];
      }
    }
    outptr_base++;
  }
}

}  // namespace depthwise
