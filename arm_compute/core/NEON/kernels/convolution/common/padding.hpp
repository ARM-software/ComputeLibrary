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

#include <cstddef>

// Utilities for copying tensor tiles and adding/removing padding.
namespace padding
{

/* Copy a tile and apply padding to the output copy.
 */
template <typename T>
void copy_and_pad_tile(
  unsigned int tile_rows,
  unsigned int tile_cols,
  unsigned int n_channels,
  const T *inptr,
  unsigned int in_row_stride,
  unsigned int in_col_stride,
  T* outptr,
  unsigned int out_row_stride,
  unsigned int out_col_stride,
  unsigned int pad_top,
  unsigned int pad_left,
  unsigned int pad_bottom,
  unsigned int pad_right,
  T pad_value=static_cast<T>(0)
);

/** Copy a tile and remove padding elements in the output.
 */
template <unsigned int TileRows, unsigned int TileCols>
class CopyCropped
{
  public:
    static void execute(
      size_t size,  // Amount of data to copy
      const void *inptr,
      size_t in_row_stride,
      size_t in_col_stride,
      void *outptr,
      size_t out_row_stride,
      size_t out_col_stride,
      unsigned int pad_top,
      unsigned int pad_left,
      unsigned int pad_bottom,
      unsigned int pad_right
    );
};

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
);

}
