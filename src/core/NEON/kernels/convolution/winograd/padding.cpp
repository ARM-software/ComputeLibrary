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
#include <cstring>
#include <cstdint>

#include "padding.hpp"

namespace padding
{

template <typename T>
void copy_and_pad_tile(
  const unsigned int tile_rows,
  const unsigned int tile_cols,
  const unsigned int n_channels,
  const T* const inptr,
  const unsigned int in_row_stride,
  const unsigned int in_col_stride,
  T* const outptr,
  const unsigned int out_row_stride,
  const unsigned int out_col_stride,
  const unsigned int pad_top,
  const unsigned int pad_left,
  const unsigned int pad_bottom,
  const unsigned int pad_right,
  const T pad_value
)
{
  for (unsigned int out_i = 0; out_i < tile_rows; out_i++)
  {
    for (unsigned int out_j = 0; out_j < tile_cols; out_j++)
    {
      T* const output = outptr + out_i*out_row_stride + out_j*out_col_stride;

      if (out_i < pad_top || tile_rows - pad_bottom <= out_i ||
          out_j < pad_left || tile_cols - pad_right <= out_j)
      {
        for (unsigned int n = 0; n < n_channels; n++)
        {
          output[n] = pad_value;
        }
      }
      else
      {
        const auto in_i = out_i - pad_top, in_j = out_j - pad_left;
        const T* const input = inptr + in_i*in_row_stride + in_j*in_col_stride;
        std::memcpy(output, input, n_channels * sizeof(T));
      }
    }
  }
}

template void copy_and_pad_tile(
  unsigned int, unsigned int, unsigned int,
  const uint8_t *, unsigned int, unsigned int,
  uint8_t *, unsigned int, unsigned int,
  unsigned int, unsigned int, unsigned int, unsigned int, uint8_t
);

template void copy_and_pad_tile(
  unsigned int, unsigned int, unsigned int,
  const float *, unsigned int, unsigned int,
  float *, unsigned int, unsigned int,
  unsigned int, unsigned int, unsigned int, unsigned int, float
);

template <unsigned int TileRows, unsigned int TileCols>
void CopyCropped<TileRows, TileCols>::execute(
  const size_t size,
  const void * const inptr,
  const size_t in_row_stride,
  const size_t in_col_stride,
  void * const outptr,
  const size_t out_row_stride,
  const size_t out_col_stride,
  const unsigned int pad_top,
  const unsigned int pad_left,
  const unsigned int pad_bottom,
  const unsigned int pad_right
)
{
  for (unsigned int out_i = 0, in_i = pad_top; in_i < TileRows - pad_bottom; out_i++, in_i++)
  {
    for (unsigned int out_j = 0, in_j = pad_left; in_j < TileCols - pad_right; out_j++, in_j++)
    {
      std::memcpy(
        static_cast<uint8_t *>(outptr) + out_i*out_row_stride + out_j*out_col_stride,
        static_cast<const uint8_t *>(inptr) + in_i*in_row_stride + in_j*in_col_stride,
        size
      );
    }
  }
}

template class CopyCropped<2, 2>;
template class CopyCropped<3, 3>;
template class CopyCropped<4, 4>;

template <typename T>
void crop_and_copy_tile(
  unsigned int tile_rows,
  unsigned int tile_cols,
  unsigned int n_channels,
  const T *inptr,
  unsigned int in_row_stride,
  unsigned int in_col_stride,
  T *outptr,
  unsigned int out_row_stride,
  unsigned int out_col_stride,
  unsigned int crop_top,
  unsigned int crop_left,
  unsigned int crop_bottom,
  unsigned int crop_right
)
{
  for (unsigned int out_i = 0, in_i = crop_top; in_i < tile_rows - crop_bottom; out_i++, in_i++)
  {
    for (unsigned int out_j = 0, in_j = crop_left; in_j < tile_cols - crop_right; out_j++, in_j++)
    {
      std::memcpy(
        outptr + out_i*out_row_stride + out_j*out_col_stride,
        inptr + in_i*in_row_stride + in_j*in_col_stride,
        sizeof(T) * n_channels
      );
    }
  }
}

template void crop_and_copy_tile(
  unsigned int tile_rows,
  unsigned int tile_cols,
  unsigned int n_channels,
  const float *inptr,
  unsigned int in_row_stride,
  unsigned int in_col_stride,
  float *outptr,
  unsigned int out_row_stride,
  unsigned int out_col_stride,
  unsigned int crop_top,
  unsigned int crop_left,
  unsigned int crop_bottom,
  unsigned int crop_right
);

}  // namespace padding
