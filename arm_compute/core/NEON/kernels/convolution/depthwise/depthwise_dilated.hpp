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

#include <deque>
#include <functional>
#include <memory>

#include "depthwise.hpp"

namespace depthwise
{

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols,
  typename TIn, typename TBias, typename TOut
>
class DilatedDepthwiseConvolution : public IDepthwiseConvolution
{
  public:
    /** Create a new dilated depthwise convolution engine.
     */
    DilatedDepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int dilation_factor,
      nck::ActivationFunction activation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

    /** Create a new dilated depthwise convolution engine.
     */
    DilatedDepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int dilation_factor, int n_output_rows, int n_output_cols,
      nck::ActivationFunction activation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

    // Cannot copy or move a DilatedDepthwiseConvolution.
    DilatedDepthwiseConvolution(DilatedDepthwiseConvolution&) = delete;
    DilatedDepthwiseConvolution operator=(DilatedDepthwiseConvolution&) = delete;

    /* Set input tensor and stride. */
    void set_input(const void *inptr) override;
    void set_input(const void *inptr, int column_stride) override;
    void set_input(const void *inptr, int row_stride, int column_stride) override;
    void set_input(const void *inptr, int batch_stride, int row_stride, int column_stride) override;

    /* Set output tensor and stride. */
    void set_output(void *outptr) override;
    void set_output(void *outptr, int column_stride) override;
    void set_output(void *outptr, int row_stride, int column_stride) override;
    void set_output(void *outptr, int batch_stride, int row_stride, int column_stride) override;

    static int get_output_size(
      int dim_size,
      unsigned int padding_before,
      unsigned int padding_after,
      int dilation_factor
    );

    int output_size(
      int dim_size, unsigned int padding_before, unsigned int padding_after
    ) const override;

    /* Weights and biases are re-ordered to improve memory access patterns. Use
     * these methods to determine the size of the re-pack buffer and to set the
     * address (and implicitly reorder the weights and biases into) the buffer.
     */
    size_t get_packed_params_size(void) const override;
    void set_packed_params_buffer(void *) override;

    void pack_params(const void *weights, const void *biases=nullptr) const override;
    void pack_params(void *buffer, const void *weights, const void *biases=nullptr) const override;
    void pack_params(
      void *buffer,
      const void* weights,
      unsigned int weight_row_stride,
      unsigned int weight_col_stride,
      const void *biases=nullptr
    ) const override;

    /* Working space is used to pad tensors on the fly. Before running any
     * inference check the amount of space required, allocate and provide a
     * pointer to the convolution engine.
     */
    size_t get_working_space_size(unsigned int nthreads=1) const override;
    void set_working_space(void *) override;

    unsigned int get_window(void) const override;
    void run(unsigned int start, unsigned int stop, unsigned int threadid=0) override;

  protected:
    /** Protected constructor which also accepts a function to construct a new
     * subconvolution
     */
    DilatedDepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int dilation_factor, int n_output_rows, int n_output_cols,
      nck::ActivationFunction activation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right,
      std::function<IDepthwiseConvolution *(int, int, int, int, int, int, nck::ActivationFunction, unsigned int, unsigned int, unsigned int, unsigned int)> subconvfn
    );

    const int _dilation_factor;
    const int _n_input_rows, _n_input_cols, _n_channels;
    const int _padding_top, _padding_left;
    const int _n_output_rows, _n_output_cols;

    /* Dilated depthwise convolution is performed through repeated calls to
     * non-dilated convolutions. If the dilation factor is $n$, then we perform
     * $(n + 1)^2$ depthwise convolutions.
     */
    using BaseDepthwise = DepthwiseConvolution<
      OutputTileRows, OutputTileCols,
      KernelRows, KernelCols,
      StrideRows, StrideCols,
      TIn, TBias, TOut
    >;
    std::deque<std::deque<std::unique_ptr<IDepthwiseConvolution>>> _convs;
};

}  // namespace depthwise
