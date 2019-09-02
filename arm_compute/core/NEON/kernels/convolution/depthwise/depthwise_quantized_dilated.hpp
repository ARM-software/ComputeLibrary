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
#include "depthwise_dilated.hpp"
#include "depthwise_quantized.hpp"

namespace depthwise {

template <unsigned int OutputTileRows, unsigned int OutputTileCols,
          unsigned int KernelRows, unsigned int KernelCols,
          unsigned int StrideRows, unsigned int StrideCols>
class QAsymm8DilatedDepthwiseConvolution
    : public DilatedDepthwiseConvolution<
          OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows,
          StrideCols, uint8_t, int32_t, uint8_t> {
public:
  /** Create a new dilated depthwise convolution engine.
   */
  QAsymm8DilatedDepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int dilation_factor, nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params &weight_quantisation,
      const qasymm8::QAsymm8Params &input_quantisation,
      const qasymm8::QAsymm8Params &output_quantisation,
      unsigned int padding_top, unsigned int padding_left,
      unsigned int padding_bottom, unsigned int padding_right);

  /** Create a new dilated depthwise convolution engine.
   */
  QAsymm8DilatedDepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int dilation_factor, int n_output_rows, int n_output_cols,
      nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params &weight_quantisation,
      const qasymm8::QAsymm8Params &input_quantisation,
      const qasymm8::QAsymm8Params &output_quantisation,
      unsigned int padding_top, unsigned int padding_left,
      unsigned int padding_bottom, unsigned int padding_right);

  /** Create a new dilated depthwise convolution engine.
   */
  QAsymm8DilatedDepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int dilation_factor, nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params &weight_quantisation,
      const qasymm8::QAsymm8Params &input_quantisation,
      const qasymm8::QAsymm8Params &output_quantisation,
      const qasymm8::QAsymm8RescaleParams &rescale_parameters,
      unsigned int padding_top, unsigned int padding_left,
      unsigned int padding_bottom, unsigned int padding_right);

  /** Create a new dilated depthwise convolution engine.
   */
  QAsymm8DilatedDepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int dilation_factor, int n_output_rows, int n_output_cols,
      nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params &weight_quantisation,
      const qasymm8::QAsymm8Params &input_quantisation,
      const qasymm8::QAsymm8Params &output_quantisation,
      const qasymm8::QAsymm8RescaleParams& rescale_parameters,
      unsigned int padding_top, unsigned int padding_left,
      unsigned int padding_bottom, unsigned int padding_right);
};

}  // namespace depthwise
