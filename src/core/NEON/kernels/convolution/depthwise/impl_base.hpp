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
#include <cstdint>
#include "depthwise.hpp"
#include "padding.hpp"
#include "utils.hpp"

#pragma once

#define MEMBERFN(TOUT) template <\
  unsigned int OutputTileRows, unsigned int OutputTileColumns,\
  unsigned int KernelRows, unsigned int KernelColumns,\
  unsigned int StrideRows, unsigned int StrideColumns,\
  typename TIn, typename TBias, typename TOut,\
  typename Derived\
> TOUT DepthwiseConvolutionBase<\
  OutputTileRows, OutputTileColumns,\
  KernelRows, KernelColumns,\
  StrideRows, StrideColumns,\
  TIn, TBias, TOut, Derived\
>

using namespace neon_convolution_kernels;

namespace depthwise
{

template <unsigned int KernelRows, unsigned int KernelColumns, size_t WeightSize, size_t BiasSize>
struct PackParameters
{
  static void execute(
    unsigned int n_channels,
    void *buffer,
    const void *weights,
    unsigned int weight_row_stride,
    unsigned int weight_col_stride,
    const void *biases
  );
};

const unsigned int CHANNEL_BLOCK = 16;

MEMBERFN(int)::get_output_size(
  const int dim_size, const unsigned int padding_before, const unsigned int padding_after
)
{
  return iceildiv(dim_size + padding_before + padding_after - KernelRows + 1, StrideRows);
}

MEMBERFN(int)::output_size(
  const int dim_size, const unsigned int padding_before, const unsigned int padding_after
) const
{
  return get_output_size(dim_size, padding_before, padding_after);
}

MEMBERFN()::DepthwiseConvolutionBase(
  const int n_batches,
  const int n_input_rows,
  const int n_input_cols,
  const int n_channels,
  ActivationFunction activation,
  const unsigned int padding_top,
  const unsigned int padding_left,
  const unsigned int padding_bottom,
  const unsigned int padding_right
) : DepthwiseConvolutionBase(
      n_batches, n_input_rows, n_input_cols, n_channels,
      get_output_size(n_input_rows, padding_top, padding_bottom),
      get_output_size(n_input_cols, padding_left, padding_right),
      activation,
      padding_top, padding_left, padding_bottom, padding_right
    )
{
}

MEMBERFN()::DepthwiseConvolutionBase(
  const int n_batches,
  const int n_input_rows,
  const int n_input_cols,
  const int n_channels,
  const int n_output_rows,
  const int n_output_cols,
  ActivationFunction activation,
  const unsigned int padding_top,
  const unsigned int padding_left,
  const unsigned int padding_bottom,
  const unsigned int padding_right
) : _input(nullptr), _output(nullptr),
    _packed_parameters(nullptr),
    _working_space(nullptr),
    _n_batches(n_batches),
    _n_input_rows(n_input_rows),
    _n_input_cols(n_input_cols),
    _n_channels(n_channels),
    _n_output_rows(n_output_rows),
    _n_output_cols(n_output_cols),
    _n_tile_rows(iceildiv(_n_output_rows, output_tile_rows)),
    _n_tile_cols(iceildiv(_n_output_cols, output_tile_cols)),
    _padding_top(padding_top),
    _padding_left(padding_left),
    _padding_bottom(padding_bottom),
    _padding_right(padding_right),
    _activation(activation),
    _input_col_stride(0), _input_row_stride(0), _input_batch_stride(0),
    _output_col_stride(0), _output_row_stride(0), _output_batch_stride(0)
{
}

MEMBERFN(void)::set_input(const void* const inptr)
{
  set_input(inptr, _n_channels);
}

MEMBERFN(void)::set_input(const void* const inptr, const int ld_col)
{
  set_input(inptr, _n_input_cols * ld_col, ld_col);
}

MEMBERFN(void)::set_input(const void* const inptr, const int ld_row, const int ld_col)
{
  set_input(inptr, _n_input_rows * ld_row, ld_row, ld_col);
}

MEMBERFN(void)::set_input(const void* const inptr, const int ld_batch, const int ld_row, const int ld_col)
{
  _input = static_cast<const TIn *>(inptr);
  _input_batch_stride = ld_batch;
  _input_row_stride = ld_row;
  _input_col_stride = ld_col;
}

MEMBERFN(void)::set_output(void* const outptr)
{
  set_output(outptr, _n_channels);
}

MEMBERFN(void)::set_output(void* const outptr, const int ld_col)
{
  set_output(outptr, _n_output_cols * ld_col, ld_col);
}

MEMBERFN(void)::set_output(void* const outptr, const int ld_row, const int ld_col)
{
  set_output(outptr, _n_output_rows * ld_row, ld_row, ld_col);
}

MEMBERFN(void)::set_output(void* const outptr, const int ld_batch, const int ld_row, const int ld_col)
{
  _output = static_cast<TOut *>(outptr);
  _output_batch_stride = ld_batch;
  _output_row_stride = ld_row;
  _output_col_stride = ld_col;
}

MEMBERFN(size_t)::get_packed_params_size(void) const
{
  return _n_channels * (sizeof(TIn)*KernelRows*KernelColumns + sizeof(TBias));
}

MEMBERFN(void)::set_packed_params_buffer(void *buffer)
{
  _packed_parameters = buffer;
}

MEMBERFN(void)::pack_params(const void *weights, const void *biases) const
{
  static_cast<const Derived *>(this)->pack_params(_packed_parameters, weights, biases);
}

MEMBERFN(void)::pack_params(void *buffer, const void *weights, const void *biases) const
{
  const unsigned int weight_col_stride = _n_channels;
  const unsigned int weight_row_stride = KernelColumns * weight_col_stride;
  static_cast<const Derived *>(this)->pack_params(
    buffer, weights, weight_row_stride, weight_col_stride, biases
  );
}

MEMBERFN(void)::pack_params(
  void * const buffer,
  const void * const weights,
  const unsigned int weight_row_stride,
  const unsigned int weight_col_stride,
  const void * const biases
) const
{
  static_cast<const Derived *>(this)->_pack_params(
    buffer, weights, weight_row_stride, weight_col_stride, biases
  );
}

MEMBERFN(void)::_pack_params(
  void * const buffer,
  const void * const weights,
  const unsigned int weight_row_stride,
  const unsigned int weight_col_stride,
  const void * const biases
) const
{
  // Default implementation
  PackParameters<KernelRows, KernelColumns, sizeof(TIn), sizeof(TOut)>::execute(
    _n_channels, buffer, weights, weight_row_stride, weight_col_stride, biases
  );
}

MEMBERFN(size_t)::get_working_space_size(const unsigned int nthreads) const
{
  return nthreads * (
    _get_input_working_space_size() + _get_output_working_space_size()
  );
}

MEMBERFN(void)::set_working_space(void *buffer)
{
  _working_space = buffer;
}

MEMBERFN(size_t)::_get_input_working_space_size(void) const
{
  return sizeof(TIn) * _n_channels;
}

MEMBERFN(size_t)::_get_output_working_space_size(void) const
{
  return sizeof(TOut) * _n_channels;
}

MEMBERFN(void *)::_get_input_working_space(const unsigned int threadid) const
{
  return static_cast<uint8_t*>(_working_space) + threadid * (
    _get_input_working_space_size() + _get_output_working_space_size()
  );
}

MEMBERFN(void *)::_get_output_working_space(const unsigned int threadid) const
{
  return static_cast<uint8_t*>(_get_input_working_space(threadid)) + _get_input_working_space_size();
}

MEMBERFN(unsigned int)::get_window() const
{
  // Parallelise over blocks of channels.
  return iceildiv(_n_channels, CHANNEL_BLOCK);
}

MEMBERFN(void)::run(
  const unsigned int start,
  const unsigned int stop,
  const unsigned int threadid
)
{
  // Clear the input padding buffer
  TIn *buf = static_cast<TIn *>(_get_input_working_space(threadid));
  const TIn pad_value = static_cast<Derived *>(this)->_input_padding_value();
  for (int n = 0; n < _n_channels; n++)
  {
    buf[n] = pad_value;
  }

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

      // Get the offset into the packed parameters
      const auto params_ptr = static_cast<const uint8_t*>(_packed_parameters) +
        start_channel*(sizeof(TIn)*KernelRows*KernelColumns + sizeof(TBias));

      // Process the row
      process_tile_row(
        threadid,
        stop_channel - start_channel,
        params_ptr,
        inptr_row + start_channel,
        outptr_row + start_channel,
        input_row_pad_top, input_pad_left, input_row_pad_bottom,
        output_row_pad_bottom,
        _n_tile_cols, _n_input_cols, _n_output_cols
      );
    }
  }
}

MEMBERFN(void)::process_tile_row(
  const unsigned int threadid,
  const int n_channels,
  const void* const packed_params,
  const TIn* const inptr,
  TOut* const outptr,
  const int row_pad_in_top,
  const int row_pad_in_left,
  const int row_pad_in_bottom,
  const int row_pad_out_bottom,
  const int n_tiles,
  const int n_input_cols,
  const int n_output_cols
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
    const TIn* const inptr_col = (inptr + ((inner_tile_cols - tile_overlap)*tile_j - col_offset)*_input_col_stride);
    TOut* const outptr_col = outptr + tile_j * output_tile_cols * _output_col_stride;

    // Process just this tile
    process_tile(
      threadid, n_channels, packed_params, inptr_col, outptr_col,
      row_pad_in_top, t_pad_in_left, row_pad_in_bottom, t_pad_in_right,  // Input paddings
      row_pad_out_bottom, t_pad_out_right  // Output paddings
    );
  }
}

MEMBERFN(TIn)::_input_padding_value(void) const
{
  return static_cast<TIn>(0);
}

MEMBERFN(void)::process_tile(
  const unsigned int threadid,
  const int n_channels,
  const void* const packed_params,
  const TIn* const inptr,
  TOut* const outptr,
  const int pad_in_top,
  const int pad_in_left,
  const int pad_in_bottom,
  const int pad_in_right,
  const int pad_out_bottom,
  const int pad_out_right
)
{
  Derived * dthis = static_cast<Derived *>(this);
  const bool pad_input = pad_in_top || pad_in_left || pad_in_bottom || pad_in_right;
  const bool pad_output = pad_out_bottom || pad_out_right;

  if (!pad_input && !pad_output)
  {
    switch(_activation)
    {
      case ActivationFunction::ReLU:
        dthis->template execute_tile<ActivationFunction::ReLU>(
          n_channels, packed_params,
          inptr, _input_row_stride, _input_col_stride,
          outptr, _output_row_stride, _output_col_stride
        );
        break;
      case ActivationFunction::ReLU6:
        dthis->template execute_tile<ActivationFunction::ReLU6>(
          n_channels, packed_params,
          inptr, _input_row_stride, _input_col_stride,
          outptr, _output_row_stride, _output_col_stride
        );
        break;
      default:
        dthis->template execute_tile<ActivationFunction::None>(
          n_channels, packed_params,
          inptr, _input_row_stride, _input_col_stride,
          outptr, _output_row_stride, _output_col_stride
        );
        break;
    }
  }
  else
  {
    // Create arrays of input and output pointers, pointing padded elements to
    // the working space padding buffers provided.
    const TIn *inptrs[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        if (i < pad_in_top || (inner_tile_rows - pad_in_bottom) <= i ||
            j < pad_in_left || (inner_tile_cols - pad_in_right) <= j)
        {
          // Padded input
          inptrs[i][j] = static_cast<const TIn *>(_get_input_working_space(threadid));
        }
        else
        {
          inptrs[i][j] = inptr + (i - pad_in_top)*_input_row_stride + (j - pad_in_left)*_input_col_stride;
        }
      }
    }

    TOut *outptrs[output_tile_rows][output_tile_cols];
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        if (i < (output_tile_rows - pad_out_bottom) &&
            j < (output_tile_cols - pad_out_right))
        {
          outptrs[i][j] = outptr + i*_output_row_stride + j*_output_col_stride;
        }
        else
        {
          outptrs[i][j] = static_cast<TOut *>(_get_output_working_space(threadid));
        }
      }
    }

    switch(_activation)
    {
      case ActivationFunction::ReLU:
        dthis->template execute_tile<ActivationFunction::ReLU>(
          n_channels, packed_params, inptrs, outptrs
        );
        break;
      case ActivationFunction::ReLU6:
        dthis->template execute_tile<ActivationFunction::ReLU6>(
          n_channels, packed_params, inptrs, outptrs
        );
        break;
      default:
        dthis->template execute_tile<ActivationFunction::None>(
          n_channels, packed_params, inptrs, outptrs
        );
        break;
    }
  }
}

MEMBERFN(int)::n_channels(void) const
{
  return _n_channels;
}

}  // namespace depthwise
