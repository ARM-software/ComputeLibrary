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

#include "depthwise_quantized_dilated.hpp"
#include "impl_dilated.hpp"

namespace depthwise {

template <unsigned int OutputTileRows, unsigned int OutputTileCols,
          unsigned int KernelRows, unsigned int KernelCols,
          unsigned int StrideRows, unsigned int StrideCols>
QAsymm8DilatedDepthwiseConvolution<OutputTileRows, OutputTileCols, KernelRows,
                                   KernelCols, StrideRows, StrideCols>::
    QAsymm8DilatedDepthwiseConvolution(
        int n_batches, int n_input_rows, int n_input_cols, int n_channels,
        int dilation_factor, nck::ActivationFunction activation,
        const qasymm8::QAsymm8Params &weight_quantisation,
        const qasymm8::QAsymm8Params &input_quantisation,
        const qasymm8::QAsymm8Params &output_quantisation,
        unsigned int padding_top, unsigned int padding_left,
        unsigned int padding_bottom, unsigned int padding_right)
    : QAsymm8DilatedDepthwiseConvolution(
          n_batches, n_input_rows, n_input_cols, n_channels, dilation_factor,
          QAsymm8DilatedDepthwiseConvolution::get_output_size(
              n_input_rows, padding_top, padding_bottom, dilation_factor),
          QAsymm8DilatedDepthwiseConvolution::get_output_size(
              n_input_cols, padding_left, padding_right, dilation_factor),
          activation, weight_quantisation, input_quantisation,
          output_quantisation, padding_top, padding_left, padding_bottom,
          padding_right) {}

template <unsigned int OutputTileRows, unsigned int OutputTileCols,
          unsigned int KernelRows, unsigned int KernelCols,
          unsigned int StrideRows, unsigned int StrideCols>
QAsymm8DilatedDepthwiseConvolution<OutputTileRows, OutputTileCols, KernelRows,
                                   KernelCols, StrideRows, StrideCols>::
    QAsymm8DilatedDepthwiseConvolution(
        int n_batches, int n_input_rows, int n_input_cols, int n_channels,
        int dilation_factor, int n_output_rows, int n_output_cols,
        nck::ActivationFunction activation,
        const qasymm8::QAsymm8Params &weight_quantisation,
        const qasymm8::QAsymm8Params &input_quantisation,
        const qasymm8::QAsymm8Params &output_quantisation,
        unsigned int padding_top, unsigned int padding_left,
        unsigned int padding_bottom, unsigned int padding_right)
    : QAsymm8DilatedDepthwiseConvolution(
          n_batches, n_input_rows, n_input_cols, n_channels, dilation_factor,
          n_output_rows, n_output_cols, activation, weight_quantisation,
          input_quantisation, output_quantisation,
          qasymm8::QAsymm8RescaleParams::make_rescale_params(
              weight_quantisation, input_quantisation, output_quantisation),
          padding_top, padding_left, padding_bottom, padding_right) {}

template <unsigned int OutputTileRows, unsigned int OutputTileCols,
          unsigned int KernelRows, unsigned int KernelCols,
          unsigned int StrideRows, unsigned int StrideCols>
QAsymm8DilatedDepthwiseConvolution<OutputTileRows, OutputTileCols, KernelRows,
                                   KernelCols, StrideRows, StrideCols>::
    QAsymm8DilatedDepthwiseConvolution(
        int n_batches, int n_input_rows, int n_input_cols, int n_channels,
        int dilation_factor, nck::ActivationFunction activation,
        const qasymm8::QAsymm8Params &weight_quantisation,
        const qasymm8::QAsymm8Params &input_quantisation,
        const qasymm8::QAsymm8Params &output_quantisation,
        const qasymm8::QAsymm8RescaleParams &rescale_parameters,
        unsigned int padding_top, unsigned int padding_left,
        unsigned int padding_bottom, unsigned int padding_right)
    : QAsymm8DilatedDepthwiseConvolution(
          n_batches, n_input_rows, n_input_cols, n_channels, dilation_factor,
          QAsymm8DilatedDepthwiseConvolution::get_output_size(
              n_input_rows, padding_top, padding_bottom, dilation_factor),
          QAsymm8DilatedDepthwiseConvolution::get_output_size(
              n_input_cols, padding_left, padding_right, dilation_factor),
          activation, weight_quantisation, input_quantisation,
          output_quantisation, rescale_parameters, padding_top, padding_left,
          padding_bottom, padding_right) {}

template <unsigned int OutputTileRows, unsigned int OutputTileCols,
          unsigned int KernelRows, unsigned int KernelCols,
          unsigned int StrideRows, unsigned int StrideCols>
QAsymm8DilatedDepthwiseConvolution<OutputTileRows, OutputTileCols, KernelRows,
                                   KernelCols, StrideRows, StrideCols>::
    QAsymm8DilatedDepthwiseConvolution(
        int n_batches, int n_input_rows, int n_input_cols, int n_channels,
        int dilation_factor, int n_output_rows, int n_output_cols,
        nck::ActivationFunction activation,
        const qasymm8::QAsymm8Params &weight_quantisation,
        const qasymm8::QAsymm8Params &input_quantisation,
        const qasymm8::QAsymm8Params &output_quantisation,
        const qasymm8::QAsymm8RescaleParams &rescale_parameters,
        unsigned int padding_top, unsigned int padding_left,
        unsigned int padding_bottom, unsigned int padding_right)
    : DilatedDepthwiseConvolution<OutputTileRows, OutputTileCols, KernelRows,
                                  KernelCols, StrideRows, StrideCols, uint8_t,
                                  int32_t, uint8_t>(
          n_batches, n_input_rows, n_input_cols, n_channels, dilation_factor,
          n_output_rows, n_output_cols, activation, padding_top, padding_left,
          padding_bottom, padding_right,
          [weight_quantisation, input_quantisation, output_quantisation,
           rescale_parameters](
              const int n_batches, const int n_input_rows,
              const int n_input_cols, const int n_channels,
              const int n_output_rows, const int n_output_cols,
              const nck::ActivationFunction activation,
              const unsigned int padding_top, const unsigned int padding_left,
              const unsigned int padding_bottom,
              const unsigned int padding_right) -> IDepthwiseConvolution * {
            return new QAsymm8DepthwiseConvolution<
                OutputTileRows, OutputTileCols, KernelRows, KernelCols,
                StrideRows, StrideCols>(
                n_batches, n_input_rows, n_input_cols, n_channels,
                n_output_rows, n_output_cols, activation, weight_quantisation,
                input_quantisation, output_quantisation, rescale_parameters,
                padding_top, padding_left, padding_bottom, padding_right);
          }) {}

} // namespace depthwise

template class depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 3, 3, 1, 1>;
template class depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 3, 3, 2, 2>;
template class depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 5, 5, 1, 1>;
template class depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 5, 5, 2, 2>;
