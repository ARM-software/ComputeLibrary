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

#include <utility>

#include "batched_blocked_gemm.hpp"
#include "winograd_gemm.hpp"

/** Example of how to construct an ACL-like interface.
 *
 * Use `get_weight_storage_size`, `get_input_storage_size` and
 * `get_output_storage_size` to allocate memory for the convolution engine.
 * Then create a `WinogradConvolutionLayer`.
 *
 * Initialise the weights using `weights_transform.run(...)`.
 *
 * For each inference:
 *   1. Transform the inputs to the Winograd domain using `input_transform.run(...)`
 *   2. Perform a number of GEMMs using `gemms.run(...)`
 *   3. Transform the output to the spatial domain using `output_transform.run(...)`
 */
template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols,
          typename TIn, typename TOut>
class WinogradConvolutionLayer
{
  private:
    const KernelShape _kernel_shape;
    const Tensor4DShape _input_shape;
    const PaddingType _padding;
    const Tensor4DShape _output_shape;
    const int _n_output_rows, _n_output_cols;
    const int _kernel_matrix_stride, _kernel_matrix_row_stride;
    const int _input_matrix_stride, _input_matrix_row_stride;
    const int _output_matrix_stride, _output_matrix_row_stride;
    const int _tile_rows, _tile_cols;
    const int _m, _k, _n;

  public:
    using WinogradBase = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using WeightsTransform = typename WinogradBase::template WeightsTransform<TIn>;
    using InputTransform = typename WinogradBase::template InputTransform<TIn>;
    using WinogradConv = typename WinogradBase::template Convolution<TOut, TIn>;
    using MultiGEMM = winograd::BatchedBlockedGemm<WinogradConv::M_BLOCK, WinogradConv::N_BLOCK, TIn, TOut>;
    using OutputTransform = typename WinogradBase::template OutputTransform<TOut>;

    /* Public member variables. */
    WeightsTransform weights_transform;  /** Operator to transform weights to Winograd domain. */
    InputTransform input_transform;      /** Operator to transform input to Winograd domain. */
    MultiGEMM gemms;                     /** Operator to perform multiple GEMMs. */
    OutputTransform output_transform;    /** Operator to transform output from Winograd domain. */

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed weights.
     *
     * @param[in]  n_output_channels Number of output feature maps.
     * @param[in]  n_input_channels  Number of input feature maps.
     */
    static unsigned int get_weight_storage_size(
      const int n_output_channels,
      const int n_input_channels
    );

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed input.
     *
     * @param[in]  n_batches  Number of batches in the input tensor.
     * @param[in]  n_channels Number of feature maps in the input tensor.
     * @param[in]  n_rows Number of rows in each feature map.
     * @param[in]  n_cols Number of columns in each feature map.
     * @param[in]  same_padding Use "SAME" padding, otherwise use "VALID".
     */
    static unsigned int get_input_storage_size(
      const int n_batches,
      const int n_channels,
      const int n_rows,
      const int n_cols,
      const bool same_padding
    );

    /** Determine how much memory (in units of TOut) to allocate for the
     * (Winograd domain) output.
     *
     * @param[in]  n_batches  Number of batches in the output tensor.
     * @param[in]  n_rows Number of rows in each feature map of the input tensor.
     * @param[in]  n_cols Number of columns in each feature map of the input tensor.
     * @param[in]  n_output_channels Number of feature maps in the output tensor.
     * @param[in]  same_padding Use "SAME" padding, otherwise use "VALID".
     */
    static unsigned int get_output_storage_size(
      const int n_batches,
      const int n_rows,
      const int n_cols,
      const int n_output_channels,
      const bool same_padding
    );

    /** Get the shape (rows, cols) of a feature map of the output tensor.
     *
     * @param[in]  n_input_rows  Number of rows in the input feature map.
     * @param[in]  n_input_cols  Number of columns in the input feature map.
     * @param[in]  same_padding Use "SAME" padding, otherwise use "VALID".
    */
    static std::pair<int, int> get_output_feature_map_shape(
      const int n_input_rows,
      const int n_input_cols,
      const bool same_padding
    );

    /** Create a new Winograd convolution layer.
     * @param[in]  n_batches Number of batches in the input and output tensors.
     * @param[in]  n_input_channels Number of feature maps in a batch of the input tensor.
     * @param[in]  n_input_rows Number of rows in a feature map of the input tensor.
     * @param[in]  n_input_cols Number of columns in a feature map of the input tensor.
     * @param[in]  n_output_channels Number of feature maps in the output tensor.
     * @param[in]  same_padding Use "SAME" padding, otherwise use "VALID".
     * @param[in]  weights Pointer to weight tensor in spatial domain. Must be ordered as "Height x Rows x Input Feature Maps x Output Feature Maps.
     * @param[out]  weights_storage Pointer to storage for weight tensor in the Winograd domain. Must be at least the size returned by `get_weight_storage_size
     * @param[in]  input Pointer to NHWC ordered input tensor, in the spatial domain.
     * @param[out] winograd_input Pointer to working space for the input tensor in the Winograd domain. Must be at least the size returned by `get_input_storage_size`.
     * @param[out]  output Pointer to NHWC ordered output tensor, in the spatial domain.
     * @param[out]  winograd_output Pointer to working space for the output tensor in the Winograd domain. Must be at least the size returned by `get_output_storage_size`.
     */
    WinogradConvolutionLayer(
      const int n_batches,
      const int n_input_channels,
      const int n_input_rows,
      const int n_input_cols,
      const int n_output_channels,
      const bool same_padding,
      const TIn* const weights,
      TIn* const weights_storage,
      const TIn* const input,
      TIn* const winograd_input,
      TOut* const output,
      TOut* const winograd_output
    );
};
