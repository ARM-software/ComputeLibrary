/*
 * Copyright (c) 2019 Arm Limited.
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

#include <limits>

#include "arm.hpp"
#include "impl_base.hpp"
#include "depthwise_quantized.hpp"

#pragma once

namespace {

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols,
  typename FInput, typename FOutput
>
static inline void tilefn_hybrid(
  int n_channels,
  const void* packed_params,
  FInput &get_input_ptr,
  FOutput &get_output_ptr,
  int32_t clamp_min,
  int32_t clamp_max,
  uint8_t input_offset,
  uint8_t output_offset
)
{
  constexpr int InnerTileRows = StrideRows * (OutputTileRows - 1) + KernelRows;
  constexpr int InnerTileCols = StrideCols * (OutputTileCols - 1) + KernelCols;

  // Offset into channels
  int channel = 0;

  // Byte type pointer to weights and biases
  const int8_t *wbptr = static_cast<const int8_t *>(packed_params);

  for (; n_channels >= 8; n_channels -= 8, channel += 8)
  {
    const int32x4_t biases[2] = {
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr)),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 4),
    };
    const int32x4_t multipliers[2] = {
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 8),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 12),
    };
    const int32x4_t shifts[2] = {
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 16),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 20),
    };
    wbptr += 24*sizeof(int32_t);

    int16x8_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        const auto w = vld1_s8(wbptr);
        weights[i][j] = reinterpret_cast<int16x8_t>(vmovl_s8(w));
        wbptr += 8;
      }
    }

    int16x8_t inputs[InnerTileRows][InnerTileCols];
    const uint8x8_t ioffset = vdup_n_u8(input_offset);
    for (unsigned int i = 0; i < InnerTileRows; i++)
    {
      for (unsigned int j = 0; j < InnerTileCols; j++)
      {
        const auto x = vld1_u8(get_input_ptr(i, j, channel));
        inputs[i][j] = reinterpret_cast<int16x8_t>(vsubl_u8(x, ioffset));
      }
    }

    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        int32x4_t accs[2];
        for (unsigned int i = 0; i < 2; i++)
        {
          accs[i] = biases[i];
        }

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            const auto w = weights[wi][wj];
            const auto x = inputs[oi * StrideRows + wi][oj * StrideCols + wj];
            accs[0] = vmlal_s16(accs[0], vget_low_s16(w), vget_low_s16(x));
            accs[1] = vmlal_s16(accs[1], vget_high_s16(w), vget_high_s16(x));
          }
        }

        int32x4_t final_accs[2];
        for (unsigned int i = 0; i < 2; i++)
        {
          const int32x4_t y = rounding_divide_by_exp2(
              saturating_doubling_high_mul(accs[i], multipliers[i]),
              shifts[i]);
          const int32x4_t offset = reinterpret_cast<int32x4_t>(vdupq_n_u32(output_offset));
          final_accs[i] = vaddq_s32(y, offset);
          final_accs[i] = vmaxq_s32(final_accs[i], vdupq_n_s32(clamp_min));
          final_accs[i] = vminq_s32(final_accs[i], vdupq_n_s32(clamp_max));
        }

        const auto elems_s16 = vuzpq_s16(vreinterpretq_s16_s32(final_accs[0]),
                                         vreinterpretq_s16_s32(final_accs[1]));
        const int8x16_t elems = vreinterpretq_s8_s16(elems_s16.val[0]);
        const uint8x8_t output =
                    vget_low_u8(vreinterpretq_u8_s8(vuzpq_s8(elems, elems).val[0]));

        vst1_u8(get_output_ptr(oi, oj, channel), output);
      }
    }
  }

  for (; n_channels; n_channels--, channel++)
  {
    // Load bias
    const int32_t bias = *reinterpret_cast<const int32_t *>(wbptr);
    const int32_t multiplier = *reinterpret_cast<const int32_t *>(wbptr + sizeof(int32_t));
    const int32_t shift = *reinterpret_cast<const int32_t *>(wbptr + 2*sizeof(int32_t));

    wbptr += 3*sizeof(int32_t);

    // Load weights
    int16_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        weights[i][j] = *(wbptr++);
      }
    }

    // Load the input activations
    int16_t inputs[InnerTileRows][InnerTileCols];
    for (unsigned int i = 0; i < InnerTileRows; i++)
    {
      for (unsigned int j = 0; j < InnerTileCols; j++)
      {
        inputs[i][j] = *(get_input_ptr(i, j, channel)) - input_offset;
      }
    }

    // Perform the convolution
    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        int32_t acc = bias;

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            const auto w = weights[wi][wj], x = inputs[oi*StrideRows + wi][oj*StrideCols + wj];
            acc += w * x;
          }
        }

        // Requantize
        acc = rounding_divide_by_exp2(
            saturating_doubling_high_mul(acc, multiplier),
            -shift);
        acc += output_offset;
        acc = std::max(acc, clamp_min);
        acc = std::min(acc, clamp_max);
        uint8_t output = static_cast<uint8_t>(acc);
        *(get_output_ptr(oi, oj, channel)) = output;
      }
    }
  }
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols,
  typename FInput, typename FOutput
>
static inline void execute_tilefn_hybrid(
  int n_channels,
  const void* packed_params,
  const ActivationFunction actfn,
  const qasymm8::QAsymm8Params &input_quant,
  const qasymm8::QAsymm8Params &output_quant,
  FInput &get_input_ptr,
  FOutput &get_output_ptr) {

  // Compute min/max clamp values
  int32_t clamp_min = std::numeric_limits<uint8_t>::min();
  int32_t clamp_max = std::numeric_limits<uint8_t>::max();

  if (actfn == ActivationFunction::ReLU) {
    clamp_min = output_quant.offset;
  }

  // Disabling Relu6 for now
  if (actfn == ActivationFunction::ReLU6) {
    const int32_t top_rail = output_quant.quantize(6.0f);
    clamp_max = std::min(clamp_max, top_rail);
  }

  // Call the tile execution method
  tilefn_hybrid<OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows,
         StrideCols>(n_channels, packed_params, get_input_ptr, get_output_ptr, clamp_min, clamp_max, input_quant.offset, output_quant.offset);
}
}



namespace depthwise {
using namespace qsymm8;
template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
QSymm8HybridPerChannelDepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::QSymm8HybridPerChannelDepthwiseConvolution(
  int n_batches, int n_input_rows, int n_input_cols, int n_channels,
  const ActivationFunction activation,
  const QSymm8PerChannelParams& weight_quantisation,
  const qasymm8::QAsymm8Params& input_quantisation,
  const qasymm8::QAsymm8Params& output_quantisation,
  unsigned int padding_top,
  unsigned int padding_left,
  unsigned int padding_bottom,
  unsigned int padding_right
) : QSymm8HybridPerChannelDepthwiseConvolution(
    n_batches, n_input_rows, n_input_cols, n_channels,
    activation, weight_quantisation, input_quantisation, output_quantisation,
    QSymm8PerChannelRescaleParams::make_rescale_params(weight_quantisation, input_quantisation, output_quantisation),
    padding_top, padding_left, padding_bottom, padding_right
  )
{
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
QSymm8HybridPerChannelDepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::QSymm8HybridPerChannelDepthwiseConvolution(
  int n_batches, int n_input_rows, int n_input_cols, int n_channels,
  const ActivationFunction activation,
  const QSymm8PerChannelParams& weight_quantisation,
  const qasymm8::QAsymm8Params& input_quantisation,
  const qasymm8::QAsymm8Params& output_quantisation,
  const QSymm8PerChannelRescaleParams& rescale_params,
  unsigned int padding_top,
  unsigned int padding_left,
  unsigned int padding_bottom,
  unsigned int padding_right
) : Base(
      n_batches, n_input_rows, n_input_cols, n_channels, activation,
      padding_top, padding_left, padding_bottom, padding_right
  ),
  _weights_quant(weight_quantisation),
  _input_quant(input_quantisation),
  _output_quant(output_quantisation),
  _rescale_parameters(rescale_params)
{
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
uint8_t QSymm8HybridPerChannelDepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::_input_padding_value(void) const
{
  return _input_quant.offset;
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
void QSymm8HybridPerChannelDepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::_pack_params(
  void * const buffer,
  const void * const weights,
  const unsigned int weight_row_stride,
  const unsigned int weight_col_stride,
  const void * const biases
) const
{
  const int8_t *wptr = static_cast<const int8_t *>(weights);
  const int32_t *bptr = static_cast<const int32_t *>(biases);
  const int32_t *mptr = static_cast<const int32_t *>(_rescale_parameters.multipliers.data());
  const int32_t *sptr = static_cast<const int32_t *>(_rescale_parameters.shifts.data());
  int8_t *outptr = static_cast<int8_t *>(buffer);

  // We set the vector length to use doubles on both Aarch64 and Aarch32.  NOTE
  // For SVE set this to half the vector length.
  unsigned int veclen = 8;

  // While there are channels left to process, pack a vector length of them at
  // a time and reduce the size of vector used as the size of the tensor
  // decreases.
  for (
    unsigned int n_channels = this->n_channels(); n_channels;
    n_channels -= veclen,
    outptr += veclen*(3*sizeof(int32_t) + this->kernel_rows*this->kernel_cols)
  )
  {
    // NOTE Ignore this section if using SVE, the vector length remains the
    // same and we just don't fill a full register for the tail.
    while (n_channels < veclen)
    {
      // Reduce the vector length to either 8 or 1 (scalar)
      // TODO Support more vector lengths in `execute_tile`.
      veclen = (veclen == 16) ? 8 : 1;
    }

    // Get pointers to bias and weight portions of the output structure.
    int32_t *out_bptr = reinterpret_cast<int32_t *>(outptr);
    int32_t *out_mptr = reinterpret_cast<int32_t *>(outptr + veclen*sizeof(int32_t));
    int32_t *out_sptr = reinterpret_cast<int32_t *>(outptr + 2*veclen*sizeof(int32_t));
    int8_t  *out_wptr = outptr + 3*veclen*sizeof(int32_t);

    // Copy a vector length of elements
    for (unsigned int n = 0; n < veclen && n < n_channels; n++)
    {
      const int32_t bias = (bptr != nullptr) ? *(bptr++) : 0;
      const int32_t multiplier = (mptr != nullptr) ? *(mptr++) : 0;
      const int32_t shift = (sptr != nullptr) ? *(sptr++) : 0;

      out_bptr[n] = bias;
      out_mptr[n] = multiplier;
      out_sptr[n] = -shift;

      for (unsigned int i = 0; i < KernelRows; i++)
      {
        int8_t *row_outptr = out_wptr + i*KernelCols*veclen;
        for (unsigned int j = 0; j < KernelCols; j++)
        {
          int8_t w = *(wptr + i*weight_row_stride + j*weight_col_stride);
          row_outptr[j*veclen + n] = w;
        }
      }
      wptr++;
    }
  }
}


template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
template <ActivationFunction Activation>
void QSymm8HybridPerChannelDepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::execute_tile(
  int n_channels,
  const void* packed_params,
  const uint8_t* inptr,
  unsigned int in_row_stride,
  unsigned int in_col_stride,
  uint8_t* outptr,
  unsigned int out_row_stride,
  unsigned int out_col_stride
) {

  // Construct methods to get pointers
  const auto get_input_ptr = [inptr, in_row_stride, in_col_stride](
      const int i, const int j, const int channel) {
    return inptr + i * in_row_stride + j * in_col_stride + channel;
  };

  const auto get_output_ptr = [outptr, out_row_stride, out_col_stride](
      const int i, const int j, const int channel) {
    return outptr + i * out_row_stride + j * out_col_stride + channel;
  };

  execute_tilefn_hybrid<OutputTileRows, OutputTileCols, KernelRows, KernelCols,
                 StrideRows, StrideCols>(
      n_channels, packed_params, Activation, _input_quant, _output_quant, get_input_ptr, get_output_ptr);
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
template <ActivationFunction Activation>
void QSymm8HybridPerChannelDepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::execute_tile(
  int n_channels,
  const void* packed_params,
  const uint8_t* inptrs[Base::inner_tile_rows][Base::inner_tile_cols],
  uint8_t* outptrs[Base::output_tile_rows][Base::output_tile_cols]
) {
  // Construct methods to get pointers
  const auto get_input_ptr = [inptrs](const int i, const int j,
                                      const int channel) {
    return inptrs[i][j] + channel;
  };

  const auto get_output_ptr = [outptrs](const int i, const int j,
                                        const int channel) {
    return outptrs[i][j] + channel;
  };

  // Call the tile execution method
  execute_tilefn_hybrid<OutputTileRows, OutputTileCols, KernelRows, KernelCols,
                 StrideRows, StrideCols>(
      n_channels, packed_params, Activation,  _input_quant, _output_quant, get_input_ptr, get_output_ptr);
}

} // namespace depthwise
