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

#include "convolution.hpp"
#include "winograd_layer.hpp"
#include "tensor.hpp"


/** Determine how much memory (in units of TIn) to allocate for the transformed
 * weights.
 */
template <
  int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols,
  typename TIn, typename TOut
>
unsigned int WinogradConvolutionLayer<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, TIn, TOut
>::get_weight_storage_size(
  const int n_output_channels,  /** Number of output feature maps. */
  const int n_input_channels    /** Number of input feature maps. */
)
{
  const KernelShape shape(
    n_output_channels, KernelRows, KernelCols, n_input_channels
  );
  return static_cast<unsigned int>(
    // WinogradConv returns the size in bytes, we divide by `sizeof(TIn)` to
    // express that in units of TIn.
    WinogradConv::get_kernel_storage_size(shape) / sizeof(TIn)
  );
}


/** Determine how much memory (in units of TIn) to allocate for the transformed
 * input.
 */
template <
  int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols,
  typename TIn, typename TOut
>
unsigned int WinogradConvolutionLayer<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, TIn, TOut
>::get_input_storage_size(
  const int n_batches,     /** Number of batches in the input tensor. */
  const int n_channels,    /** Number of feature maps in the input tensor. */
  const int n_rows,        /** Number of rows in each feature map. */
  const int n_cols,        /** Number of columns in each feature map. */
  const bool same_padding  /** Use "SAME" padding, otherwise use "VALID". */
)
{
  // Construct shapes for the input and kernel tensors.
  const Tensor4DShape input_shape(n_batches, n_rows, n_cols, n_channels);
  const KernelShape kern_shape(1, KernelRows, KernelCols, n_channels);
  const PaddingType padding = (same_padding) ? PADDING_SAME : PADDING_VALID;

  // Return the size, converted into units of TIn
  return static_cast<unsigned int>(
    WinogradConv::get_input_storage_size(kern_shape, input_shape, padding) /
    sizeof(TIn)
  );
}


/** Determine how much memory (in units of TOut) to allocate for the (Winograd
 * domain) output.
 */
template <
  int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols,
  typename TIn, typename TOut
>
unsigned int WinogradConvolutionLayer<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, TIn, TOut
>::get_output_storage_size(
  const int n_batches,          /** Number of batches in the output tensor. */
  const int n_rows,             /** Number of rows in each feature map of the input tensor. */
  const int n_cols,             /** Number of columns in each feature map of the input tensor. */
  const int n_output_channels,  /** Number of feature maps in the output tensor. */
  const bool same_padding       /** Use "SAME" padding, otherwise use "VALID". */
)
{
  // Construct shapes for the input and kernel tensors.
  const Tensor4DShape input_shape(n_batches, n_rows, n_cols, 1);
  const KernelShape kern_shape(n_output_channels, KernelRows, KernelCols, 1);
  const PaddingType padding = (same_padding) ? PADDING_SAME : PADDING_VALID;

  // Return the size, converted into units of TOut
  return static_cast<unsigned int>(
    WinogradConv::get_output_storage_size(kern_shape, input_shape, padding) /
    sizeof(TOut)
  );
}


/** Get the shape (rows, cols) of a feature map of the output tensor. */
template <
  int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols,
  typename TIn, typename TOut
>
std::pair<int, int> WinogradConvolutionLayer<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, TIn, TOut
>::get_output_feature_map_shape(
  const int n_input_rows,  /** Number of rows in the input feature map. */
  const int n_input_cols,  /** Number of columns in the input feature map. */
  const bool same_padding  /** Use "SAME" padding, otherwise use "VALID". */
)
{
  // Construct shapes for the input and kernel tensors.
  const Tensor4DShape input_shape(1, n_input_rows, n_input_cols, 1);
  const KernelShape kern_shape(1, KernelRows, KernelCols, 1);
  const PaddingType padding = (same_padding) ? PADDING_SAME : PADDING_VALID;

  // Compute the new shape
  const auto output_shape = WinogradConv::get_output_shape(
    kern_shape, input_shape, padding
  );

  return std::make_pair(output_shape.n_rows, output_shape.n_cols);
}


/** Create a new Winograd convolution layer.
 */
template <
  int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols,
  typename TIn, typename TOut
>
WinogradConvolutionLayer<OutputTileRows, OutputTileCols, KernelRows, KernelCols, TIn, TOut>::
WinogradConvolutionLayer(
  const int n_batches,          /** Number of batches in the input and output tensors. */
  const int n_input_channels,   /** Number of feature maps in a batch of the input tensor. */
  const int n_input_rows,       /** Number of rows in a feature map of the input tensor. */
  const int n_input_cols,       /** Number of columns in a feature map of the input tensor. */
  const int n_output_channels,  /** Number of feature maps in the output tensor. */
  const bool same_padding,      /** Use "SAME" padding, otherwise use "VALID". */
  const TIn* const weights,     /** Pointer to weight tensor in spatial domain. Must be ordered as "Height x Rows x Input Feature Maps x Output Feature Maps. */
  TIn* const winograd_weights,  /** Pointer to storage for weight tensor in the Winograd domain. Must be at least the size returned by `get_weight_storage_size`. */
  const TIn* const input,       /** Pointer to NHWC ordered input tensor, in the spatial domain. */
  TIn* const winograd_input,    /** Pointer to working space for the input tensor in the Winograd domain. Must be at least the size returned by `get_input_storage_size`. */
  TOut* const output,           /** Pointer to NHWC ordered output tensor, in the spatial domain. */
  TOut* const winograd_output   /** Pointer to working space for the output tensor in the Winograd domain. Must be at least the size returned by `get_output_storage_size`. */
) : _kernel_shape(n_output_channels, KernelRows, KernelCols, n_input_channels),
    _input_shape(n_batches, n_input_rows, n_input_cols, n_input_channels),
    _padding(same_padding ? PADDING_SAME : PADDING_VALID),
    _output_shape(WinogradConv::get_output_shape(_kernel_shape, _input_shape, _padding)),
    _n_output_rows(_output_shape.n_rows),
    _n_output_cols(_output_shape.n_cols),
    _kernel_matrix_stride(WinogradConv::get_kernel_matrix_stride(_kernel_shape)),
    _kernel_matrix_row_stride(roundup(n_output_channels, WinogradConv::N_BLOCK)),
    _input_matrix_stride(WinogradConv::get_input_matrix_stride(_kernel_shape, _input_shape, _padding)),
    _input_matrix_row_stride(n_input_channels),
    _output_matrix_stride(WinogradConv::get_output_matrix_stride(_kernel_shape, _input_shape, _padding)),
    _output_matrix_row_stride(_kernel_matrix_row_stride),
    _tile_rows(iceildiv(_n_output_rows, OutputTileRows)),
    _tile_cols(iceildiv(_n_output_cols, OutputTileCols)),
    _m(n_batches * _tile_rows * _tile_cols),
    _k(n_input_channels),
    _n(n_output_channels),
    weights_transform(
      weights, winograd_weights,
      _kernel_matrix_stride, _kernel_matrix_row_stride,
      n_output_channels, n_input_channels
    ),
    input_transform(
      input, n_batches, n_input_rows, n_input_cols, n_input_channels, _padding,
      winograd_input, _input_matrix_stride, _input_matrix_row_stride
    ),
    gemms(
      WinogradBase::N_GEMMS, _m, _k, _n,
      _input_matrix_stride, _input_matrix_row_stride,
      _kernel_matrix_stride, _kernel_matrix_row_stride,
      _output_matrix_stride, _output_matrix_row_stride,
      winograd_input, winograd_weights, winograd_output
    ),
    output_transform(
      winograd_output, _output_matrix_stride, _output_matrix_row_stride,
      output, n_batches, _n_output_rows, _n_output_cols, n_output_channels
    )
{
}

// Instantiate valid implementations.
template class WinogradConvolutionLayer<2, 2, 3, 3, float, float>;
template class WinogradConvolutionLayer<4, 4, 3, 3, float, float>;
