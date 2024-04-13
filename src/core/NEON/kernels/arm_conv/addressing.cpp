/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "addressing.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cstring>

namespace arm_conv {
namespace addressing {

void fill_pointer_array(
  size_t element_size,
  void **dest_raw, const unsigned int array_rows, const unsigned int array_cols,
  void *base_ptr_raw, size_t ld_row, size_t ld_col,
  void *pad_buffer_raw,
  const unsigned int pad_top, const unsigned int valid_rows,
  const unsigned int pad_left, const unsigned int valid_cols
)
{
  auto dest = reinterpret_cast<char **>(dest_raw);
  auto base_ptr = reinterpret_cast<char *>(base_ptr_raw);
  auto pad_buffer = reinterpret_cast<char *>(pad_buffer_raw);
  ld_row *= element_size;
  ld_col *= element_size;

  const auto last_valid_row = std::min(pad_top + valid_rows, array_rows);
  const auto last_valid_col = std::min(pad_left + valid_cols, array_cols);

  unsigned int i = 0;
  for (; i < pad_top; i++)
  {
    for (unsigned int j = 0; j < array_cols; j++)
    {
      *(dest++) = pad_buffer;
    }
  }
  for (; i < last_valid_row; i++)
  {
    unsigned int j = 0;
    auto colptr = base_ptr;
    base_ptr += ld_row;

    for (; j < pad_left; j++)
    {
      *(dest++) = pad_buffer;
    }
    for (; j < last_valid_col; j++)
    {
      *(dest++) = colptr;
      colptr += ld_col;
    }
    for (; j < array_cols; j++)
    {
      *(dest++) = pad_buffer;
    }
  }
  for (; i < array_rows; i++)
  {
    for (unsigned int j = 0; j < array_cols; j++)
    {
      *(dest++) = pad_buffer;
    }
  }
}


void fill_pointer_array_generic_kernel(
  const size_t element_size,
  void **dest_raw,
  const unsigned int output_rows, const unsigned int output_cols,
  const unsigned int kernel_rows, const unsigned int kernel_cols,
  const unsigned int stride_rows, const unsigned int stride_cols,
  void *base_ptr_raw, size_t ld_row, size_t ld_col,
  void *pad_buffer_raw,
  const unsigned int pad_top, const unsigned int valid_rows,
  const unsigned int pad_left, const unsigned int valid_cols
)
{
  auto dest = reinterpret_cast<char **>(dest_raw);
  auto base_ptr = reinterpret_cast<char *>(base_ptr_raw);
  auto pad_buffer = reinterpret_cast<char *>(pad_buffer_raw);
  ld_row *= element_size;
  ld_col *= element_size;

  const auto last_valid_row = pad_top + valid_rows;
  const auto last_valid_col = pad_left + valid_cols;
  const auto point_stride = output_rows * output_cols;

  // Iterate over the output points, after every point increment the pointer
  // into the address array.
  for (unsigned int oi = 0; oi < output_rows; oi++)
  {
    for (unsigned int oj = 0; oj < output_cols; oj++)
    {
      auto point_dest = dest;
      dest++;

      // Iterate over kernel points and fill in the pointer array.
      unsigned int ki = 0, ii = oi*stride_rows;
      for (; ii < pad_top && ki < kernel_rows; ii++, ki++)
      {
        // Fill with padding
        for (unsigned int j = 0; j < kernel_cols; j++)
        {
          *point_dest = pad_buffer;
          point_dest += point_stride;
        }
      }
      for (; ii < last_valid_row && ki < kernel_rows; ii++, ki++)
      {
        unsigned int kj = 0, ij = oj*stride_cols;
        for (; ij < pad_left && kj < kernel_cols; ij++, kj++)
        {
          // Padding
          *point_dest = pad_buffer;
          point_dest += point_stride;
        }
        for (; ij < last_valid_col && kj < kernel_cols; ij++, kj++)
        {
          *point_dest = base_ptr + (ii - pad_top)*ld_row + (ij - pad_left)*ld_col;
          point_dest += point_stride;
        }
        for (; kj < kernel_cols; kj++)
        {
          // Padding
          *point_dest = pad_buffer;
          point_dest += point_stride;
        }
      }
      for (; ki < kernel_rows; ki++)
      {
        // Fill with padding
        for (unsigned int j = 0; j < kernel_cols; j++)
        {
          *point_dest = pad_buffer;
          point_dest += point_stride;
        }
      }
    }
  }
}

/* Patch array constructor
 *
 * Some depthwise kernels require an NCHW-ordered patch of input. Here we
 * construct such a patch, and fill in an array of pointers to the rows of the
 * patch.
 */
void fill_nchw_patch_array(
  size_t element_size,
  const void **dest_row_pointers_raw,  // Array of pointers to each row of the patch
  void *dest_patch_raw,  // Pointer to space which can be used to construct the patch
  const unsigned int patch_rows, unsigned int patch_cols,  // Patch size
  const void *src_ptr_raw, size_t ld_row, size_t ld_col,  // Source tensor
  const void *pad_row,  // Pointer to a row of padding values
  const unsigned int pad_top, const unsigned int valid_rows,
  const unsigned int pad_left, const unsigned int valid_cols
)
{
  // Convert into more useful types
  auto row_pointers = reinterpret_cast<const char **>(dest_row_pointers_raw);
  auto dest_patch = reinterpret_cast<char *>(dest_patch_raw);
  auto src = reinterpret_cast<const char *>(src_ptr_raw);
  ld_row *= element_size;
  ld_col *= element_size;

  // Round up the patch columns to be a full quad
  patch_cols = arm_gemm::roundup<unsigned int>(patch_cols, 16 / element_size);

  const auto last_valid_row = std::min(pad_top + valid_rows, patch_rows);
  const auto last_valid_col = std::min(pad_left + valid_cols, patch_cols);

  // Construct the patch and row pointer array together
  unsigned int i = 0;
  for (; i < pad_top; i++)
  {
    // Insert pointers into the padding row
    *(row_pointers++) = reinterpret_cast<const char *>(pad_row);
  }
  for (; i < last_valid_row; i++)
  {
    // Get a copy of the pointer for this row
    auto colptr = src;
    src += ld_row;

    // If the input is already in NCHW format (ld_col == element_size) AND
    // there is no padding, then we just use a pointer to the source tensor;
    // otherwise we need to construct a patch and provide a pointer to it.
    if (ld_col == element_size && pad_left == 0 && last_valid_col == patch_cols)
    {
      *(row_pointers++) = colptr;
    }
    else
    {
      auto patch_col = dest_patch;
      *(row_pointers++) = dest_patch;
      dest_patch += element_size * patch_cols;  // Move the patch pointer on

      // Construct the patch; fill the entirety with padding and then copy in
      // the valid elements.
      memcpy(patch_col, pad_row, element_size * patch_cols);
      patch_col += pad_left * element_size;  // Move over the left padding

      if (ld_col == element_size)
      {
        // If the input is NCHW then copy across as many columns as we can.
        memcpy(patch_col, colptr, (last_valid_col - pad_left) * element_size);
      }
      else
      {
        // If the input is NHWC then copy columns across in turn.
        for (auto j = pad_left; j < last_valid_col; j++)
        {
          memcpy(patch_col, colptr, element_size);  // Copy the valid element
          patch_col += element_size;  // Progress the patch destination
          colptr += ld_col;  // Progress the patch source
        }
      }
    }
  }
  for (; i < patch_rows; i++)
  {
    // Insert pointers into the padding row
    *(row_pointers++) = reinterpret_cast<const char *>(pad_row);
  }
}


/* Patch array constructor (generic kernels)
 *
 * Construct an array of pointers; one pointer for each output row for each
 * kernel point. Pointers should point at a whole number of QUADS containing an
 * input point for each output point. If the kernel column stride is 1 and the
 * data is NCHW then the input tensor might be addressed directly, otherwise a
 * new patch sample might need to be constructed.
 */
void fill_patch_array_generic_kernel(
  size_t element_size,
  const void **dest_pointers_raw,  // Pointers: one per output row per kernel point
  void *patch_raw,  // Pointer to space which can be used to construct the patch
  const unsigned int output_rows, const unsigned int output_cols,
  const unsigned int kernel_rows, const unsigned int kernel_cols,
  const unsigned int stride_rows, const unsigned int stride_cols,
  const void *src_ptr_raw, size_t ld_row, size_t ld_col,  // Source tensor
  const void *pad_row,  // Pointer to a row of padding values
  const unsigned int pad_top, const unsigned int valid_rows,
  const unsigned int pad_left, const unsigned int valid_cols
)
{
  auto dest = reinterpret_cast<const char **>(dest_pointers_raw);
  auto patch = reinterpret_cast<char *>(patch_raw);
  auto src_ptr = reinterpret_cast<const char *>(src_ptr_raw);
  ld_row *= element_size;
  ld_col *= element_size;

  // Round up the patch columns to a multiple of quad-length
  const auto patch_cols = arm_gemm::roundup<unsigned int>(output_cols, 16 / element_size);

  const auto input_rows = kernel_rows + (output_rows - 1) * stride_rows;
  const auto last_valid_row = std::min(pad_top + valid_rows, input_rows);

  const auto input_cols = kernel_cols + (output_cols - 1) * stride_cols;
  const auto last_valid_col = std::min(pad_left + valid_cols, input_cols);

  for (auto ki = 0u; ki < kernel_rows; ki++)
  {
    for (auto kj = 0u; kj < kernel_cols; kj++)
    {
      auto oi = 0u, ii = ki;
      for (; oi < output_rows && ii < pad_top; oi++, ii += stride_rows)
      {
        // Insert a pointer to the padding row
        *(dest++) = reinterpret_cast<const char *>(pad_row);
      }
      for (; oi < output_rows && ii < last_valid_row; oi++, ii += stride_rows)
      {
        auto rowptr = src_ptr + (ii - pad_top) * ld_row;

        // Construct a sample of the input here
        auto patch_pos = patch;
        *(dest++) = patch;
        patch += patch_cols * element_size;

        // Fill with padding
        memcpy(patch_pos, pad_row, patch_cols * element_size);

        // Fill in the valid elements
        auto oj = 0u, ij = kj;
        for (; oj < patch_cols && ij < pad_left; oj++, ij += stride_cols)
        {
          // Do nothing for padding
          patch_pos += element_size;
        }
        for (; oj < patch_cols && ij < last_valid_col; oj++, ij += stride_cols)
        {
          // Copy from the source tensor
          memcpy(patch_pos, rowptr + (ij - pad_left)*ld_col, element_size);
          patch_pos += element_size;
        }
        // No action required for right-hand padding
      }
      for (; oi < output_rows; oi++)
      {
        *(dest++) = reinterpret_cast<const char *>(pad_row);
      }
    }
  }
}

}  // namespace addressing
}  // namespace arm_conv
