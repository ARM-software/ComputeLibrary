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

#pragma once
#include "depthwise.hpp"
#include "qasymm8.hpp"

namespace depthwise
{

namespace nck = neon_convolution_kernels;

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
class QAsymm8DepthwiseConvolution : public DepthwiseConvolutionBase<
  OutputTileRows, OutputTileCols,
  KernelRows, KernelCols,
  StrideRows, StrideCols,
  uint8_t, int32_t, uint8_t,
  QAsymm8DepthwiseConvolution<OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols>
>
{
  using Base = DepthwiseConvolutionBase<
    OutputTileRows, OutputTileCols,
    KernelRows, KernelCols,
    StrideRows, StrideCols,
    uint8_t, int32_t, uint8_t,
    QAsymm8DepthwiseConvolution<OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols>
  >;
  friend Base;
  using InputType = typename Base::InputType;
  using OutputType = typename Base::OutputType;

  public:
    QAsymm8DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params& weight_quantisation,
      const qasymm8::QAsymm8Params& input_quantisation,
      const qasymm8::QAsymm8Params& output_quantisation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

    QAsymm8DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int n_output_rows, int n_output_cols,
      nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params& weight_quantisation,
      const qasymm8::QAsymm8Params& input_quantisation,
      const qasymm8::QAsymm8Params& output_quantisation,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

    QAsymm8DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params& weight_quantisation,
      const qasymm8::QAsymm8Params& input_quantisation,
      const qasymm8::QAsymm8Params& output_quantisation,
      const qasymm8::QAsymm8RescaleParams& rescale_parameters,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

    QAsymm8DepthwiseConvolution(
      int n_batches, int n_input_rows, int n_input_cols, int n_channels,
      int n_output_rows, int n_output_cols,
      nck::ActivationFunction activation,
      const qasymm8::QAsymm8Params& weight_quantisation,
      const qasymm8::QAsymm8Params& input_quantisation,
      const qasymm8::QAsymm8Params& output_quantisation,
      const qasymm8::QAsymm8RescaleParams& rescale_parameters,
      unsigned int padding_top,
      unsigned int padding_left,
      unsigned int padding_bottom,
      unsigned int padding_right
    );

  protected:
    uint8_t _input_padding_value(void) const;

    void _pack_params(
      void *buffer,
      const void *weights,
      unsigned int weight_row_stride,
      unsigned int weight_col_stride,
      const void *biases=nullptr
    ) const;

    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const uint8_t* inptr,
      unsigned int in_row_stride,
      unsigned int in_col_stride,
      uint8_t* outptr,
      unsigned int out_row_stride,
      unsigned int out_col_stride
    );

    template <nck::ActivationFunction Activation>
    void execute_tile(
      int n_channels,
      const void* packed_params,
      const uint8_t* inptrs[Base::inner_tile_rows][Base::inner_tile_cols],
      uint8_t* outptrs[Base::output_tile_rows][Base::output_tile_cols]
    );

  private:
    // Quantization parameters
    const qasymm8::QAsymm8Params _weights_quant, _inputs_quant, _output_quant;
    const qasymm8::QAsymm8RescaleParams rescale_parameters;
};

}  // namespace depthwise
