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
 * The above copyright notice and this permission notice shall be included in
 * all
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

#include "depthwise_dilated.hpp"
#include "utils.hpp"

#define MEMBERFN(TOUT)                                                         \
  template <unsigned int OutputTileRows, unsigned int OutputTileColumns,       \
            unsigned int KernelRows, unsigned int KernelColumns,               \
            unsigned int StrideRows, unsigned int StrideColumns, typename TIn, \
            typename TBias, typename TOut>                                     \
  TOUT DilatedDepthwiseConvolution<OutputTileRows, OutputTileColumns,          \
                                   KernelRows, KernelColumns, StrideRows,      \
                                   StrideColumns, TIn, TBias, TOut>

namespace depthwise {

MEMBERFN()
::DilatedDepthwiseConvolution(const int n_batches, const int n_input_rows,
                              const int n_input_cols, const int n_channels,
                              const int dilation_factor,
                              nck::ActivationFunction activation,
                              const unsigned int padding_top,
                              const unsigned int padding_left,
                              const unsigned int padding_bottom,
                              const unsigned int padding_right)
    : DilatedDepthwiseConvolution(
          n_batches, n_input_rows, n_input_cols, n_channels, dilation_factor,
          DilatedDepthwiseConvolution::get_output_size(
              n_input_rows, padding_top, padding_bottom, dilation_factor),
          DilatedDepthwiseConvolution::get_output_size(
              n_input_cols, padding_left, padding_right, dilation_factor),
          activation, padding_top, padding_left, padding_bottom,
          padding_right) {}

MEMBERFN()
::DilatedDepthwiseConvolution(const int n_batches, const int n_input_rows,
                              const int n_input_cols, const int n_channels,
                              const int dilation_factor,
                              const int n_output_rows, const int n_output_cols,
                              nck::ActivationFunction activation,
                              const unsigned int padding_top,
                              const unsigned int padding_left,
                              const unsigned int, // padding_bottom
                              const unsigned int  // padding_right
                              )
    : DilatedDepthwiseConvolution(
          n_batches, n_input_rows, n_input_cols, n_channels, dilation_factor,
          n_output_rows, n_output_cols, activation, padding_top, padding_left,
          0, 0,
          // Function which creates a new (standard) depthwise convolution
          [](const int n_batches, const int n_input_rows,
             const int n_input_cols, const int n_channels,
             const int n_output_rows, const int n_output_cols,
             const nck::ActivationFunction activation,
             const unsigned int padding_top, const unsigned int padding_left,
             const unsigned int padding_bottom,
             const unsigned int padding_right) -> IDepthwiseConvolution * {
            return new DepthwiseConvolution<
                OutputTileRows, OutputTileColumns, KernelRows, KernelColumns,
                StrideRows, StrideColumns, TIn, TBias, TOut>(
                n_batches, n_input_rows, n_input_cols, n_channels,
                n_output_rows, n_output_cols, activation, padding_top,
                padding_left, padding_bottom, padding_right);
          }) {}

MEMBERFN()
::DilatedDepthwiseConvolution(
    const int n_batches, const int n_input_rows, const int n_input_cols,
    const int n_channels, const int dilation_factor, const int n_output_rows,
    const int n_output_cols, nck::ActivationFunction activation,
    const unsigned int padding_top, const unsigned int padding_left,
    const unsigned int, // padding_bottom
    const unsigned int, // padding_right
    std::function<IDepthwiseConvolution *(
        int, int, int, int, int, int, nck::ActivationFunction, unsigned int,
        unsigned int, unsigned int, unsigned int)>
        subconvfn // Function to create a new convolution
    )
    : _dilation_factor(dilation_factor), _n_input_rows(n_input_rows),
      _n_input_cols(n_input_cols), _n_channels(n_channels),
      _padding_top(static_cast<int>(padding_top)),
      _padding_left(static_cast<int>(padding_left)),
      _n_output_rows(n_output_rows), _n_output_cols(n_output_cols),
      _convs(_dilation_factor) {
  // Instantiate the base convolutions
  for (int i = 0; i < _dilation_factor; i++) {
    // Compute properties of this row of base convolutions
    const int row_top =
        i * StrideRows - _padding_top; // -ve values are in the padding
    const int row_pad_top =
        row_top < 0 ? iceildiv(-row_top, dilation_factor) : 0;

    const int _n_input_rows = iceildiv(n_input_rows - i, dilation_factor);
    const int _n_output_rows = iceildiv(n_output_rows - i, dilation_factor);

    for (int j = 0; j < _dilation_factor; j++) {
      // Compute properties of the base convolution
      const int col_left =
          j * StrideColumns - padding_left; // -ve values are in the padding
      const int col_pad_left =
          col_left < 0 ? iceildiv(-col_left, dilation_factor) : 0;

      const int _n_input_cols = iceildiv(n_input_cols - j, dilation_factor);
      const int _n_output_cols = iceildiv(n_output_cols - j, dilation_factor);

      // Create new depthwise convolution engine and include it in the vector
      // of engines. The new depthwise convolution engine is created by calling
      // the delegate function we received as an argument.
      _convs[i].emplace_back(subconvfn(
          n_batches, _n_input_rows, _n_input_cols, n_channels, _n_output_rows,
          _n_output_cols, activation,
          // Note: since we have computed the output tensor size we don't need
          // to explicitly provide bottom and right padding values to the
          // depthwise convolution.
          row_pad_top, col_pad_left, 0, 0));
    }
  }
}

MEMBERFN(void)::set_input(const void *const inptr) {
  set_input(inptr, _n_channels);
}

MEMBERFN(void)::set_input(const void *const inptr, const int ldcol) {
  set_input(inptr, _n_input_cols * ldcol, ldcol);
}

MEMBERFN(void)
::set_input(const void *const inptr, const int ldrow, const int ldcol) {
  set_input(inptr, _n_input_rows * ldrow, ldrow, ldcol);
}

MEMBERFN(void)
::set_input(const void *const inptr, const int ldbatch, const int ldrow,
            const int ldcol) {
  // Compute dilated strides
  const int ldrow_dilated = ldrow * _dilation_factor;
  const int ldcol_dilated = ldcol * _dilation_factor;

  // Pass input parameters on to base convolutions
  for (int i = 0; i < _dilation_factor; i++) {
    const int top_pos =
        i * StrideRows - _padding_top +
        ((static_cast<int>(i * StrideRows) < _padding_top)
             ? iceildiv(_padding_top - i * StrideRows, _dilation_factor) *
                   _dilation_factor
             : 0);
    const TIn *const inptr_i =
        static_cast<const TIn *>(inptr) + top_pos * ldrow;

    for (int j = 0; j < _dilation_factor; j++) {
      int left_pos = j * StrideColumns - _padding_left;
      while (left_pos < 0)
        left_pos += _dilation_factor;

      // Modify the pointer to point to the first element of the dilated input
      // tensor, then set the input for this convolution engine.
      const void *const inptr_ij = inptr_i + left_pos * ldcol;
      _convs[i][j]->set_input(inptr_ij, ldbatch, ldrow_dilated, ldcol_dilated);
    }
  }
}

MEMBERFN(void)::set_output(void *const outptr) {
  set_output(outptr, _n_channels);
}

MEMBERFN(void)::set_output(void *const outptr, const int ldcol) {
  set_output(outptr, _n_output_cols * ldcol, ldcol);
}

MEMBERFN(void)
::set_output(void *const outptr, const int ldrow, const int ldcol) {
  set_output(outptr, _n_output_rows * ldrow, ldrow, ldcol);
}

MEMBERFN(void)
::set_output(void *const outptr, const int ldbatch, const int ldrow,
             const int ldcol) {
  // Compute dilated strides
  const int ldrow_dilated = ldrow * _dilation_factor;
  const int ldcol_dilated = ldcol * _dilation_factor;

  // Pass input parameters on to base convolutions
  for (int i = 0; i < _dilation_factor; i++) {
    for (int j = 0; j < _dilation_factor; j++) {
      // Modify the pointer to point to the first element of the dilated input
      // tensor, then set the input for this convolution engine.
      void *const outptr_ij =
          static_cast<TOut *>(outptr) + i * ldrow + j * ldcol;
      _convs[i][j]->set_output(outptr_ij, ldbatch, ldrow_dilated,
                               ldcol_dilated);
    }
  }
}

MEMBERFN(int)
::get_output_size(const int dim_size, const unsigned int padding_before,
                  const unsigned int padding_after, const int dilation_factor) {
  const int input_size =
      dim_size + static_cast<int>(padding_before + padding_after);
  const int window_size = (KernelRows - 1) * dilation_factor + 1;
  return iceildiv(input_size - window_size + 1, StrideRows);
}

MEMBERFN(int)
::output_size(const int dim_size, const unsigned int padding_before,
              const unsigned int padding_after) const {
  return get_output_size(dim_size, padding_before, padding_after,
                         _dilation_factor);
}

MEMBERFN(size_t)::get_packed_params_size(void) const {
  return _convs[0][0]->get_packed_params_size();
}

MEMBERFN(void)::set_packed_params_buffer(void *buffer) {
  // Set the buffer for all convolution engines
  for (auto &&row : _convs) {
    for (auto &&conv : row) {
      conv->set_packed_params_buffer(buffer);
    }
  }
}

MEMBERFN(void)
::pack_params(const void *const weights, const void *const biases) const {
  _convs[0][0]->pack_params(weights, biases);
}

MEMBERFN(void)
::pack_params(void *const buffer, const void *const weights,
              const void *const biases) const {
  _convs[0][0]->pack_params(buffer, weights, biases);
}

MEMBERFN(void)
::pack_params(void *const buffer, const void *const weights,
              const unsigned int ldrow, const unsigned int ldcol,
              const void *const biases) const {
  _convs[0][0]->pack_params(buffer, weights, ldrow, ldcol, biases);
}

MEMBERFN(size_t)::get_working_space_size(unsigned int nthreads) const {
  return _convs[0][0]->get_working_space_size(nthreads);
}

MEMBERFN(void)::set_working_space(void *const ws) {
  // Use the same working space set for all contained depthwise engines.
  for (auto &&row : _convs) {
    for (auto &&conv : row) {
      conv->set_working_space(ws);
    }
  }
}

MEMBERFN(unsigned int)::get_window(void) const {
  return _convs[0][0]->get_window();
}

MEMBERFN(void)
::run(const unsigned int start, const unsigned int stop,
      const unsigned int threadid) {
  // Run each contained convolution in turn
  for (auto &&row : _convs) {
    for (auto &&conv : row) {
      conv->run(start, stop, threadid);
    }
  }
}

} // namespace depthwise
