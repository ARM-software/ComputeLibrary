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

#include "impl_base.hpp"

// TODO Move to common utilities somewhere
template <size_t Size> struct DType { };
template <> struct DType<1> { using scalar_type = uint8_t; };
template <> struct DType<2> { using scalar_type = uint16_t; };
template <> struct DType<4> { using scalar_type = uint32_t; };

namespace depthwise
{

template <unsigned int KernelRows, unsigned int KernelColumns, size_t WeightSize, size_t BiasSize>
void PackParameters<KernelRows, KernelColumns, WeightSize, BiasSize>::execute(
  unsigned int n_channels,
  void *buffer,
  const void *weights,
  const unsigned int weight_row_stride,
  const unsigned int weight_col_stride,
  const void *biases
)
{
  using TWeight = typename DType<WeightSize>::scalar_type;
  using TBias = typename DType<BiasSize>::scalar_type;

  auto buffer_ptr = static_cast<uint8_t *>(buffer);
  auto weights_ptr = static_cast<const TWeight *>(weights);
  auto biases_ptr = static_cast<const TBias *>(biases);

  const unsigned int veclen = 16 / WeightSize;
  for (; n_channels >= veclen; n_channels -= veclen)
  {
    // Copy biases
    for (unsigned int i = 0; i < veclen; i++)
    {
      auto ptr = reinterpret_cast<TBias *>(buffer_ptr);
      *ptr = (biases_ptr == nullptr) ? 0x0 : *(biases_ptr++);
      buffer_ptr += BiasSize;
    }

    // Copy weights
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelColumns; j++)
      {
        for (unsigned int c = 0; c < veclen; c++)
        {
          *(reinterpret_cast<TWeight *>(buffer_ptr)) = weights_ptr[i*weight_row_stride + j*weight_col_stride + c];
          buffer_ptr += WeightSize;
        }
      }
    }
    weights_ptr += veclen;
  }
  for (; n_channels; n_channels--)
  {
    // Copy bias
    auto ptr = reinterpret_cast<TBias *>(buffer_ptr);
    *ptr = (biases_ptr == nullptr) ? 0x0 : *(biases_ptr++);
    buffer_ptr += BiasSize;

    // Copy weights
    for (unsigned int i = 0; i < KernelRows; i++)
    {
      for (unsigned int j = 0; j < KernelColumns; j++)
      {
        *(reinterpret_cast<TWeight *>(buffer_ptr)) = weights_ptr[i*weight_row_stride + j*weight_col_stride];
        buffer_ptr += WeightSize;
      }
    }
    weights_ptr++;
  }
}

template struct PackParameters<3, 3, 2ul, 2ul>;
template struct PackParameters<3, 3, 4ul, 4ul>;
template struct PackParameters<5, 5, 2ul, 2ul>;
template struct PackParameters<5, 5, 4ul, 4ul>;
}  // namespace
