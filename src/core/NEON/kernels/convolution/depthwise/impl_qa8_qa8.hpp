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

/*
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 *          NOTE: Header to be included by implementation files only.
 *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

#include <limits>

#include "arm_compute/core/NEON/kernels/convolution/common/arm.hpp"
#include "arm_compute/core/NEON/kernels/convolution/depthwise/impl_base.hpp"
#include "arm_compute/core/NEON/kernels/convolution/depthwise/depthwise_quantized.hpp"

#pragma once

// Comment the following to use floating-point based quantisation, leave
// uncommented to use fixed-point.
#define FIXED_POINT_REQUANTISATION 1

using namespace neon_convolution_kernels;
using namespace qasymm8;

template <typename T>
struct clamp_to_limits
{
  template <typename U>
  static inline U clamp(const U& v)
  {
    const std::numeric_limits<T> limits;
    const U min = static_cast<U>(limits.min());
    const U max = static_cast<U>(limits.max());
    return std::min(std::max(v, min), max);
  }

  template <typename U>
  static inline T clamp_and_cast(const U& v)
  {
    return static_cast<U>(clamp(v));
  }
};

template <typename T>
inline T saturating_doubling_high_mul(const T&, const int32_t&);

template <>
inline int32x4_t saturating_doubling_high_mul(const int32x4_t& a, const int32_t& b)
{
  return vqrdmulhq_n_s32(a, b);
}

template <>
inline int32_t saturating_doubling_high_mul(const int32_t& a, const int32_t& b)
{
  return vget_lane_s32(vqrdmulh_n_s32(vdup_n_s32(a), b), 0);
}

template <typename T>
inline T rounding_divide_by_exp2(const T& x, const int exponent);

template <>
inline int32x4_t rounding_divide_by_exp2(const int32x4_t& x, const int exponent)
{
  const int32x4_t shift = vdupq_n_s32(-exponent);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, shift), 31);
  const int32x4_t fixed = vqaddq_s32(x, fixup);
  return vrshlq_s32(fixed, shift);
}

template <>
inline int32x2_t rounding_divide_by_exp2(const int32x2_t& x, const int exponent)
{
  const int32x2_t shift = vdup_n_s32(-exponent);
  const int32x2_t fixup = vshr_n_s32(vand_s32(x, shift), 31);
  const int32x2_t fixed = vqadd_s32(x, fixup);
  return vrshl_s32(fixed, shift);
}

template <>
inline int32_t rounding_divide_by_exp2(const int32_t& x, const int exponent)
{
  const int32x2_t xs = vdup_n_s32(x);
  return vget_lane_s32(rounding_divide_by_exp2(xs, exponent), 0);
}

namespace depthwise
{
template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
QAsymm8DepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::QAsymm8DepthwiseConvolution(
  int n_batches, int n_input_rows, int n_input_cols, int n_channels,
  const ActivationFunction activation,
  const QAsymm8Params& weight_quantisation,
  const QAsymm8Params& input_quantisation,
  const QAsymm8Params& output_quantisation,
  unsigned int padding_top,
  unsigned int padding_left,
  unsigned int padding_bottom,
  unsigned int padding_right
) : QAsymm8DepthwiseConvolution(
    n_batches, n_input_rows, n_input_cols, n_channels,
    activation, weight_quantisation, input_quantisation, output_quantisation,
    QAsymm8RescaleParams::make_rescale_params(weight_quantisation, input_quantisation, output_quantisation),
    padding_top, padding_left, padding_bottom, padding_right
  )
{
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
QAsymm8DepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::QAsymm8DepthwiseConvolution(
  int n_batches, int n_input_rows, int n_input_cols, int n_channels,
  const ActivationFunction activation,
  const QAsymm8Params& weight_quantisation,
  const QAsymm8Params& input_quantisation,
  const QAsymm8Params& output_quantisation,
  const QAsymm8RescaleParams& rescale_params,
  unsigned int padding_top,
  unsigned int padding_left,
  unsigned int padding_bottom,
  unsigned int padding_right
) : Base(
    n_batches, n_input_rows, n_input_cols, n_channels,
    get_activation_fn(activation, output_quantisation),
    padding_top, padding_left, padding_bottom, padding_right
  ),
  _weights_quant(weight_quantisation),
  _inputs_quant(input_quantisation),
  _output_quant(output_quantisation),
  rescale_parameters(rescale_params)
{
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
ActivationFunction QAsymm8DepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::get_activation_fn(
  const ActivationFunction activation,
  const QAsymm8Params& output_quant
)
{
  if (
    (activation == ActivationFunction::ReLU &&
     output_quant.quantize(0) == 0) ||
    (activation == ActivationFunction::ReLU6 &&
     output_quant.quantize(0) == 0 &&
     output_quant.dequantize(255) <= 6.0f)
  )
  {
    // If the range of values which can be represented by a quantized value are
    // within the range that would be produced by the activation function, then
    // the activation function is redundant and can be skipped.
    return ActivationFunction::None;
  }
  else if(
    activation == ActivationFunction::ReLU6 &&
    output_quant.dequantize(255) <= 6.0f
  )
  {
    // If the largest value that can be represented by a quantized value is
    // lower than the upper boundary, then the activation function can be
    // relaxed to a ReLU.
    return ActivationFunction::ReLU;
  }

  return activation;
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
uint8_t QAsymm8DepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::_input_padding_value(void) const
{
  return _inputs_quant.offset;
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
void QAsymm8DepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::_pack_params(
  void * const buffer,
  const void * const weights,
  const unsigned int weight_row_stride,
  const unsigned int weight_col_stride,
  const void * const biases
) const
{
  const uint8_t *wptr = static_cast<const uint8_t *>(weights);
  const int32_t *bptr = static_cast<const int32_t *>(biases);
  uint8_t *outptr = static_cast<uint8_t *>(buffer);

  // We set the vector length to use quad registers on Aarch64 and only doubles
  // on Aarch32. NOTE For SVE set this to the actual vector length.
#if defined(__aarch64__)
  unsigned int veclen = 16;
#else
#if defined(__arm__)
  unsigned int veclen = 8;
#endif
#endif

  // Compute the rank 0 offset arising from the quantisation parameters.
  const int32_t rank0_offset = (KernelRows * KernelCols *
                                static_cast<int32_t>(_weights_quant.offset) *
                                static_cast<int32_t>(_inputs_quant.offset));

  // While there are channels left to process, pack a vector length of them at
  // a time and reduce the size of vector used as the size of the tensor
  // decreases.
  for (
    unsigned int n_channels = this->n_channels(); n_channels;
    n_channels -= veclen,
    outptr += veclen*(sizeof(int32_t) + this->kernel_rows*this->kernel_cols)
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
    uint8_t *out_wptr = outptr + veclen*sizeof(int32_t);

    // Copy a vector length of elements
    for (unsigned int n = 0; n < veclen && n < n_channels; n++)
    {
      int32_t bias = (bptr != nullptr) ? *(bptr++) : 0;
      uint32_t weight_sum = 0;

      for (unsigned int i = 0; i < KernelRows; i++)
      {
        uint8_t *row_outptr = out_wptr + i*KernelCols*veclen;
        for (unsigned int j = 0; j < KernelCols; j++)
        {
          uint8_t w = *(wptr + i*weight_row_stride + j*weight_col_stride);
          row_outptr[j*veclen + n] = w;
          weight_sum += static_cast<uint32_t>(w);
        }
      }
      wptr++;

      // Include in the bias contributions from the quantisation offset
      int32_t rank1_offset = static_cast<int32_t>(
        static_cast<uint32_t>(_inputs_quant.offset) * weight_sum
      );
      out_bptr[n] = bias + rank0_offset - rank1_offset;
    }
  }
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
template<ActivationFunction Activation>
void QAsymm8DepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::execute_tile(
  int n_channels,
  const void* packed_params,
  const uint8_t* inptr,
  const unsigned int in_row_stride,
  const unsigned int in_col_stride,
  uint8_t* outptr,
  const unsigned int out_row_stride,
  const unsigned int out_col_stride
)
{
  // Activation parameters (unused if Activation is None)
  const uint8_t aqmin = _output_quant.offset;
  const uint8_t aqmax = (Activation == ActivationFunction::ReLU6) ?
    std::min<uint8_t>(255u, _output_quant.quantize(6.0f)) : 255u;

  // Byte type pointer to weights and biases
  const uint8_t *wbptr = static_cast<const uint8_t *>(packed_params);

#if defined(__aarch64__)  // Under Aarch64 only use quad registers
  for (; n_channels >= 16; n_channels -= 16)
  {
    // Load biases
    const int32x4_t biases[4] = {
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr)),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 4),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 8),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 12)
    };
    wbptr += 16*sizeof(int32_t);

    // Load weights
    uint8x16_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        weights[i][j] = vld1q_u8(wbptr);
        wbptr += 16;
      }
    }

    // Load the input activations
    uint8x16_t inputs[Base::inner_tile_rows][Base::inner_tile_cols];
    for (unsigned int i = 0; i < Base::inner_tile_rows; i++)
    {
      for (unsigned int j = 0; j < Base::inner_tile_cols; j++)
      {
        inputs[i][j] = vld1q_u8(inptr + i*in_row_stride + j*in_col_stride);
      }
    }
    inptr += 16;

    // Perform the convolution
    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        // Two sets of operations are required, we perform the
        // multiply-accumulates for the convolution proper but must also sum
        // the tile elements to account for the _weight_ offset.
        uint32x4_t accs[4];
        for (unsigned int i = 0; i < 4; i++)
        {
          accs[i] = reinterpret_cast<uint32x4_t>(biases[i]);
        }

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            // Get relevant weight and activation pixel
            const uint8x16_t w = weights[wi][wj];
            const uint8x16_t x = inputs[oi*StrideRows + wi][oj*StrideCols + wj];

            // Perform multiplication and accumulation
            const uint16x8_t muls[2] = {
              vmull_u8(vget_low_u8(w), vget_low_u8(x)),
              vmull_u8(vget_high_u8(w), vget_high_u8(x))
            };

            const uint8x8_t woffset = vdup_n_u8(_weights_quant.offset);
            const uint16x8_t sum_elems[2] = {
              vmull_u8(vget_low_u8(x), woffset),
              vmull_u8(vget_high_u8(x), woffset)
            };

            const uint32x4_t tmps[4] = {
              vsubl_u16(vget_low_u16(muls[0]), vget_low_u16(sum_elems[0])),
              vsubl_u16(vget_high_u16(muls[0]), vget_high_u16(sum_elems[0])),
              vsubl_u16(vget_low_u16(muls[1]), vget_low_u16(sum_elems[1])),
              vsubl_u16(vget_high_u16(muls[1]), vget_high_u16(sum_elems[1])),
            };
            for (unsigned int i = 0; i < 4; i++)
            {
              accs[i] = vaddq_u32(accs[i], tmps[i]);
            }
          }
        }

        // Rescale the accumulator and add in the new offset.
        uint32x4_t final_accs[4];
        for (unsigned int i = 0; i < 4; i++)
        {
#ifdef FIXED_POINT_REQUANTISATION
          const int32x4_t y = rounding_divide_by_exp2(
            saturating_doubling_high_mul(
              reinterpret_cast<int32x4_t>(accs[i]), rescale_parameters.multiplier
            ),
            rescale_parameters.shift
          );
          const int32x4_t offset = reinterpret_cast<int32x4_t>(vdupq_n_u32(_output_quant.offset));
          final_accs[i] = reinterpret_cast<uint32x4_t>(vmaxq_s32(vaddq_s32(y, offset), vdupq_n_s32(0)));
#else  // floating point requantisation
          float32x4_t fp_acc = vcvtq_f32_s32(reinterpret_cast<int32x4_t>(accs[i]));
          fp_acc = vmulq_f32(fp_acc, vdupq_n_f32(rescale_parameters.rescale));
          fp_acc = vaddq_f32(fp_acc, vdupq_n_f32(static_cast<float>(_output_quant.offset)));
          fp_acc = vmaxq_f32(fp_acc, vdupq_n_f32(0.0f));
          final_accs[i] = vcvtq_u32_f32(fp_acc);
#endif
        }

        uint8x16_t output = vcombine_u8(
          vqmovn_u16(vcombine_u16(vqmovn_u32(final_accs[0]), vqmovn_u32(final_accs[1]))),
          vqmovn_u16(vcombine_u16(vqmovn_u32(final_accs[2]), vqmovn_u32(final_accs[3])))
        );

        // Apply the activation function
        if (Activation == ActivationFunction::ReLU ||
            Activation == ActivationFunction::ReLU6)
        {
          output = vmaxq_u8(output, vdupq_n_u8(aqmin));
        }
        if (Activation == ActivationFunction::ReLU6)
        {
          output = vminq_u8(output, vdupq_n_u8(aqmax));
        }

        vst1q_u8(outptr + oi*out_row_stride + oj*out_col_stride, output);
      }
    }
    outptr += 16;
  }
#endif  // defined(__aarch64__)
  for (; n_channels >= 8; n_channels -= 8)
  {
    const int32x4_t biases[2] = {
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr)),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 4),
    };
    wbptr += 8*sizeof(int32_t);

    uint8x8_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        weights[i][j] = vld1_u8(wbptr);
        wbptr += 8;
      }
    }

    uint8x8_t inputs[Base::inner_tile_rows][Base::inner_tile_cols];
    for (unsigned int i = 0; i < Base::inner_tile_rows; i++)
    {
      for (unsigned int j = 0; j < Base::inner_tile_cols; j++)
      {
        inputs[i][j] = vld1_u8(inptr + i*in_row_stride + j*in_col_stride);
      }
    }
    inptr += 8;

    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        uint32x4_t accs[2];
        for (unsigned int i = 0; i < 2; i++)
        {
          accs[i] = reinterpret_cast<uint32x4_t>(biases[i]);
        }

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            const uint8x8_t w = weights[wi][wj];
            const uint8x8_t x = inputs[oi*StrideRows + wi][oj*StrideCols + wj];

            const uint16x8_t muls = vmull_u8(w, x);
            const uint8x8_t woffset = vdup_n_u8(_weights_quant.offset);
            const uint16x8_t sum_elems = vmull_u8(x, woffset);

            const uint32x4_t tmps[2] = {
              vsubl_u16(vget_low_u16(muls), vget_low_u16(sum_elems)),
              vsubl_u16(vget_high_u16(muls), vget_high_u16(sum_elems)),
            };
            for (unsigned int i = 0; i < 2; i++)
            {
              accs[i] = vaddq_u32(accs[i], tmps[i]);
            }
          }
        }

        uint32x4_t final_accs[2];
        for (unsigned int i = 0; i < 2; i++)
        {
#ifdef FIXED_POINT_REQUANTISATION
          const int32x4_t y = rounding_divide_by_exp2(
            saturating_doubling_high_mul(
              reinterpret_cast<int32x4_t>(accs[i]), rescale_parameters.multiplier
            ),
            rescale_parameters.shift
          );
          const int32x4_t offset = reinterpret_cast<int32x4_t>(vdupq_n_u32(_output_quant.offset));
          final_accs[i] = reinterpret_cast<uint32x4_t>(vmaxq_s32(vaddq_s32(y, offset), vdupq_n_s32(0)));
#else  // floating point requantisation
          float32x4_t fp_acc = vcvtq_f32_s32(reinterpret_cast<int32x4_t>(accs[i]));
          fp_acc = vmulq_f32(fp_acc, vdupq_n_f32(rescale_parameters.rescale));
          fp_acc = vaddq_f32(fp_acc, vdupq_n_f32(static_cast<float>(_output_quant.offset)));
          fp_acc = vmaxq_f32(fp_acc, vdupq_n_f32(0.0f));
          final_accs[i] = vcvtq_u32_f32(fp_acc);
#endif
        }

        uint8x8_t output = vqmovn_u16(vcombine_u16(vqmovn_u32(final_accs[0]), vqmovn_u32(final_accs[1])));

        // Apply the activation function
        if (Activation == ActivationFunction::ReLU ||
            Activation == ActivationFunction::ReLU6)
        {
          output = vmax_u8(output, vdup_n_u8(aqmin));
        }
        if (Activation == ActivationFunction::ReLU6)
        {
          output = vmin_u8(output, vdup_n_u8(aqmax));
        }

        vst1_u8(outptr + oi*out_row_stride + oj*out_col_stride, output);
      }
    }
    outptr += 8;
  }
  for (; n_channels; n_channels--)
  {
    // Load bias
    const int32_t bias = *reinterpret_cast<const int32_t *>(wbptr);
    wbptr += sizeof(int32_t);

    // Load weights
    uint8_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        weights[i][j] = *(wbptr++);
      }
    }

    // Load the input activations
    uint8_t inputs[Base::inner_tile_rows][Base::inner_tile_cols];
    for (unsigned int i = 0; i < Base::inner_tile_rows; i++)
    {
      for (unsigned int j = 0; j < Base::inner_tile_cols; j++)
      {
        inputs[i][j] = *(inptr + i*in_row_stride + j*in_col_stride);
      }
    }
    inptr++;

    // Perform the convolution
    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        int32_t acc = bias;
        uint32_t element_sum = 0;

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            const auto w = weights[wi][wj], x = inputs[oi*StrideRows + wi][oj*StrideCols + wj];
            acc += static_cast<int32_t>(static_cast<uint32_t>(w) * static_cast<uint32_t>(x));
            element_sum += static_cast<uint32_t>(x);
          }
        }

        acc -= static_cast<int32_t>(element_sum) * static_cast<int32_t>(_weights_quant.offset);

        // Requantize
#ifdef FIXED_POINT_REQUANTISATION
        acc = rounding_divide_by_exp2(
            saturating_doubling_high_mul(acc, rescale_parameters.multiplier),
            rescale_parameters.shift
        );
        acc += _output_quant.offset;
        uint8_t output = clamp_to_limits<uint8_t>::clamp_and_cast<int32_t>(acc);
#else  // floating point requantization
        float fp_acc = static_cast<float>(acc);
        fp_acc *= rescale_parameters.rescale;
        fp_acc += static_cast<float>(_output_quant.offset);
        fp_acc = std::max<float>(fp_acc, 0.0f);
        uint8_t output = static_cast<uint8_t>(std::min<int32_t>(static_cast<int32_t>(fp_acc), 255));
#endif

        // Apply the activation function
        if (Activation == ActivationFunction::ReLU ||
            Activation == ActivationFunction::ReLU6)
        {
          output = std::max(output, aqmin);
        }
        if (Activation == ActivationFunction::ReLU6)
        {
          output = std::min(output, aqmax);
        }

        *(outptr + oi*out_row_stride + oj*out_col_stride) = output;
      }
    }
    outptr++;
  }
}

template <
  unsigned int OutputTileRows, unsigned int OutputTileCols,
  unsigned int KernelRows, unsigned int KernelCols,
  unsigned int StrideRows, unsigned int StrideCols
>
template<ActivationFunction Activation>
void QAsymm8DepthwiseConvolution<
  OutputTileRows, OutputTileCols, KernelRows, KernelCols, StrideRows, StrideCols
>::execute_tile(
  int n_channels,
  const void* packed_params,
  const uint8_t* inptrs[Base::inner_tile_rows][Base::inner_tile_cols],
  uint8_t* outptrs[Base::output_tile_rows][Base::output_tile_cols]
)
{
  // Activation parameters (unused if Activation is None)
  const uint8_t aqmin = _output_quant.offset;
  const uint8_t aqmax = (Activation == ActivationFunction::ReLU6) ?
    std::min<uint8_t>(255u, _output_quant.quantize(6.0f)) : 255u;

  // Byte type pointer to weights and biases
  const uint8_t *wbptr = static_cast<const uint8_t *>(packed_params);

  // Offset into input/output tensors
  int n = 0;

#if defined(__aarch64__)  // Under Aarch64 only use quad registers
  for (; n_channels >= 16; n_channels -= 16, n += 16)
  {
    // Load biases
    const int32x4_t biases[4] = {
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr)),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 4),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 8),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 12)
    };
    wbptr += 16*sizeof(int32_t);

    // Load weights
    uint8x16_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        weights[i][j] = vld1q_u8(wbptr);
        wbptr += 16;
      }
    }

    // Load the input activations
    uint8x16_t inputs[Base::inner_tile_rows][Base::inner_tile_cols];
    for (unsigned int i = 0; i < Base::inner_tile_rows; i++)
    {
      for (unsigned int j = 0; j < Base::inner_tile_cols; j++)
      {
        inputs[i][j] = vld1q_u8(inptrs[i][j] + n);
      }
    }

    // Perform the convolution
    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        // Two sets of operations are required, we perform the
        // multiply-accumulates for the convolution proper but must also sum
        // the tile elements to account for the _weight_ offset.
        uint32x4_t accs[4];
        for (unsigned int i = 0; i < 4; i++)
        {
          accs[i] = reinterpret_cast<uint32x4_t>(biases[i]);
        }

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            // Get relevant weight and activation pixel
            const uint8x16_t w = weights[wi][wj];
            const uint8x16_t x = inputs[oi*StrideRows + wi][oj*StrideCols + wj];

            // Perform multiplication and accumulation
            const uint16x8_t muls[2] = {
              vmull_u8(vget_low_u8(w), vget_low_u8(x)),
              vmull_u8(vget_high_u8(w), vget_high_u8(x))
            };

            const uint8x8_t woffset = vdup_n_u8(_weights_quant.offset);
            const uint16x8_t sum_elems[2] = {
              vmull_u8(vget_low_u8(x), woffset),
              vmull_u8(vget_high_u8(x), woffset)
            };

            const uint32x4_t tmps[4] = {
              vsubl_u16(vget_low_u16(muls[0]), vget_low_u16(sum_elems[0])),
              vsubl_u16(vget_high_u16(muls[0]), vget_high_u16(sum_elems[0])),
              vsubl_u16(vget_low_u16(muls[1]), vget_low_u16(sum_elems[1])),
              vsubl_u16(vget_high_u16(muls[1]), vget_high_u16(sum_elems[1])),
            };
            for (unsigned int i = 0; i < 4; i++)
            {
              accs[i] = vaddq_u32(accs[i], tmps[i]);
            }
          }
        }

        // Rescale the accumulator and add in the new offset.
        uint32x4_t final_accs[4];
        for (unsigned int i = 0; i < 4; i++)
        {
#ifdef FIXED_POINT_REQUANTISATION
          const int32x4_t y = rounding_divide_by_exp2(
            saturating_doubling_high_mul(
              reinterpret_cast<int32x4_t>(accs[i]), rescale_parameters.multiplier
            ),
            rescale_parameters.shift
          );
          const int32x4_t offset = reinterpret_cast<int32x4_t>(vdupq_n_u32(_output_quant.offset));
          final_accs[i] = reinterpret_cast<uint32x4_t>(vmaxq_s32(vaddq_s32(y, offset), vdupq_n_s32(0)));
#else  // floating point requantisation
          float32x4_t fp_acc = vcvtq_f32_s32(reinterpret_cast<int32x4_t>(accs[i]));
          fp_acc = vmulq_f32(fp_acc, vdupq_n_f32(rescale_parameters.rescale));
          fp_acc = vaddq_f32(fp_acc, vdupq_n_f32(static_cast<float>(_output_quant.offset)));
          fp_acc = vmaxq_f32(fp_acc, vdupq_n_f32(0.0f));
          final_accs[i] = vcvtq_u32_f32(fp_acc);
#endif
        }

        uint8x16_t output = vcombine_u8(
          vqmovn_u16(vcombine_u16(vqmovn_u32(final_accs[0]), vqmovn_u32(final_accs[1]))),
          vqmovn_u16(vcombine_u16(vqmovn_u32(final_accs[2]), vqmovn_u32(final_accs[3])))
        );

        // Apply the activation function
        if (Activation == ActivationFunction::ReLU ||
            Activation == ActivationFunction::ReLU6)
        {
          output = vmaxq_u8(output, vdupq_n_u8(aqmin));
        }
        if (Activation == ActivationFunction::ReLU6)
        {
          output = vminq_u8(output, vdupq_n_u8(aqmax));
        }

        vst1q_u8(outptrs[oi][oj] + n, output);
      }
    }
  }
#endif  // defined(__aarch64__)
  for (; n_channels >= 8; n_channels -= 8, n += 8)
  {
    const int32x4_t biases[2] = {
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr)),
      vld1q_s32(reinterpret_cast<const int32_t *>(wbptr) + 4),
    };
    wbptr += 8*sizeof(int32_t);

    uint8x8_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        weights[i][j] = vld1_u8(wbptr);
        wbptr += 8;
      }
    }

    uint8x8_t inputs[Base::inner_tile_rows][Base::inner_tile_cols];
    for (unsigned int i = 0; i < Base::inner_tile_rows; i++)
    {
      for (unsigned int j = 0; j < Base::inner_tile_cols; j++)
      {
        inputs[i][j] = vld1_u8(inptrs[i][j] + n);
      }
    }

    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        uint32x4_t accs[2];
        for (unsigned int i = 0; i < 2; i++)
        {
          accs[i] = reinterpret_cast<uint32x4_t>(biases[i]);
        }

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            const uint8x8_t w = weights[wi][wj];
            const uint8x8_t x = inputs[oi*StrideRows + wi][oj*StrideCols + wj];

            const uint16x8_t muls = vmull_u8(w, x);
            const uint8x8_t woffset = vdup_n_u8(_weights_quant.offset);
            const uint16x8_t sum_elems = vmull_u8(x, woffset);

            const uint32x4_t tmps[2] = {
              vsubl_u16(vget_low_u16(muls), vget_low_u16(sum_elems)),
              vsubl_u16(vget_high_u16(muls), vget_high_u16(sum_elems)),
            };
            for (unsigned int i = 0; i < 2; i++)
            {
              accs[i] = vaddq_u32(accs[i], tmps[i]);
            }
          }
        }

        uint32x4_t final_accs[2];
        for (unsigned int i = 0; i < 2; i++)
        {
#ifdef FIXED_POINT_REQUANTISATION
          const int32x4_t y = rounding_divide_by_exp2(
            saturating_doubling_high_mul(
              reinterpret_cast<int32x4_t>(accs[i]), rescale_parameters.multiplier
            ),
            rescale_parameters.shift
          );
          const int32x4_t offset = reinterpret_cast<int32x4_t>(vdupq_n_u32(_output_quant.offset));
          final_accs[i] = reinterpret_cast<uint32x4_t>(vmaxq_s32(vaddq_s32(y, offset), vdupq_n_s32(0)));
#else  // floating point requantisation
          float32x4_t fp_acc = vcvtq_f32_s32(reinterpret_cast<int32x4_t>(accs[i]));
          fp_acc = vmulq_f32(fp_acc, vdupq_n_f32(rescale_parameters.rescale));
          fp_acc = vaddq_f32(fp_acc, vdupq_n_f32(static_cast<float>(_output_quant.offset)));
          fp_acc = vmaxq_f32(fp_acc, vdupq_n_f32(0.0f));
          final_accs[i] = vcvtq_u32_f32(fp_acc);
#endif
        }

        uint8x8_t output = vqmovn_u16(vcombine_u16(vqmovn_u32(final_accs[0]), vqmovn_u32(final_accs[1])));

        // Apply the activation function
        if (Activation == ActivationFunction::ReLU ||
            Activation == ActivationFunction::ReLU6)
        {
          output = vmax_u8(output, vdup_n_u8(aqmin));
        }
        if (Activation == ActivationFunction::ReLU6)
        {
          output = vmin_u8(output, vdup_n_u8(aqmax));
        }

        vst1_u8(outptrs[oi][oj] + n, output);
      }
    }
  }
  for (; n_channels; n_channels--, n++)
  {
    // Load bias
    const int32_t bias = *reinterpret_cast<const int32_t *>(wbptr);
    wbptr += sizeof(int32_t);

    // Load weights
    uint8_t weights[KernelRows][KernelCols];
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelCols; j++)
      {
        weights[i][j] = *(wbptr++);
      }
    }

    // Load the input activations
    uint8_t inputs[Base::inner_tile_rows][Base::inner_tile_cols];
    for (unsigned int i = 0; i < Base::inner_tile_rows; i++)
    {
      for (unsigned int j = 0; j < Base::inner_tile_cols; j++)
      {
        inputs[i][j] = *(inptrs[i][j] + n);
      }
    }

    // Perform the convolution
    for (unsigned int oi = 0; oi < OutputTileRows; oi++)
    {
      for (unsigned int oj = 0; oj < OutputTileCols; oj++)
      {
        int32_t acc = bias;
        uint32_t element_sum = 0;

        for (unsigned int wi = 0; wi < KernelRows; wi++)
        {
          for (unsigned int wj = 0; wj < KernelCols; wj++)
          {
            const auto w = weights[wi][wj], x = inputs[oi*StrideRows + wi][oj*StrideCols + wj];
            acc += static_cast<int32_t>(static_cast<uint32_t>(w) * static_cast<uint32_t>(x));
            element_sum += static_cast<uint32_t>(x);
          }
        }

        acc -= static_cast<int32_t>(element_sum) * static_cast<int32_t>(_weights_quant.offset);

        // Requantize
#ifdef FIXED_POINT_REQUANTISATION
        acc = rounding_divide_by_exp2(
            saturating_doubling_high_mul(acc, rescale_parameters.multiplier),
            rescale_parameters.shift
        );
        acc += _output_quant.offset;
        uint8_t output = clamp_to_limits<uint8_t>::clamp_and_cast<int32_t>(acc);
#else  // floating point requantization
        float fp_acc = static_cast<float>(acc);
        fp_acc *= rescale_parameters.rescale;
        fp_acc += static_cast<float>(_output_quant.offset);
        fp_acc = std::max<float>(fp_acc, 0.0f);
        uint8_t output = static_cast<uint8_t>(std::min<int32_t>(static_cast<int32_t>(fp_acc), 255));
#endif

        // Apply the activation function
        if (Activation == ActivationFunction::ReLU ||
            Activation == ActivationFunction::ReLU6)
        {
          output = std::max(output, aqmin);
        }
        if (Activation == ActivationFunction::ReLU6)
        {
          output = std::min(output, aqmax);
        }

        *(outptrs[oi][oj] + n) = output;
      }
    }
  }
}

}  // namespace depthwise
