/*
 * Copyright (c) 2022 Arm Limited.
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

#include "src/core/NEON/kernels/assembly/winograd.hpp"
#include <algorithm>
#include <functional>

namespace arm_conv {
namespace winograd {
namespace weight_transform {

/* Driver class for the Winograd weight transforms.
 */
template <typename TIn, typename TOut=TIn>
class Transform : public ITransform
{
  using Kernel = std::function<void(
    unsigned int n_channels,  // Number of channels to transform
    const TIn *inptr, size_t ld_in_row, size_t ld_in_col,
    TOut *outptr, size_t ld_out_matrix
  )>;

  const std::string m_name;
  const unsigned int m_kernel_rows, m_kernel_cols;
  const unsigned int m_transformed_tile_rows, m_transformed_tile_cols;
  const Kernel m_kernel;

  void execute_internal(
    const ConvolutionArgs &args,
    const TIn *inptr, size_t ld_in_row, size_t ld_in_col, size_t ld_input_channel,
    TOut *outptr, size_t ld_out_matrix, size_t ld_out_row,
    unsigned int thread_id, unsigned int n_threads
  ) const
  {
    // Stripe groups of input channels over threads, this should reduce false
    // sharing of the output matrix.
    constexpr auto n_input_channels_per_thread = 16u;

    // Get the initial offset for the input and output pointers
    const auto offset = thread_id * n_input_channels_per_thread;
    inptr += offset * ld_input_channel;
    outptr += offset * ld_out_row;

    for (auto start_ic = thread_id * n_input_channels_per_thread;
         start_ic < args.n_input_channels;
         start_ic += n_threads * n_input_channels_per_thread)
    {
      // Now iterate over the input channels assigned to this thread.
      const auto end_ic = std::min(args.n_input_channels,
                                   start_ic + n_input_channels_per_thread);
      for (auto ic = start_ic; ic < end_ic; ic++)
      {
        m_kernel(args.n_output_channels, inptr, ld_in_row, ld_in_col,
                 outptr, ld_out_matrix);
        inptr += ld_input_channel;
        outptr += ld_out_row;
      }

      // Progress the pointers to the account for the work not performed by
      // this thread.
      const auto skip = (n_threads - 1) * n_input_channels_per_thread;
      inptr += skip * ld_input_channel;
      outptr += skip * ld_out_row;
    }
  }

  public:
  Transform(
    const std::string &name,
    unsigned int kernel_rows, unsigned int kernel_cols,
    unsigned int transformed_tile_rows, unsigned int transformed_tile_cols,
    const Kernel kernel
  )
  : m_name(name),
    m_kernel_rows(kernel_rows), m_kernel_cols(kernel_cols),
    m_transformed_tile_rows(transformed_tile_rows), m_transformed_tile_cols(transformed_tile_cols),
    m_kernel(kernel)
  {
  }

  const std::string &get_name(void) const override { return m_name; }

  unsigned int get_kernel_rows(void) const override { return m_kernel_rows; }
  unsigned int get_kernel_cols(void) const override { return m_kernel_cols; }

  unsigned int get_transformed_tile_rows(void) const override { return m_transformed_tile_rows; }
  unsigned int get_transformed_tile_cols(void) const override { return m_transformed_tile_cols; }

  void execute(
    const ConvolutionArgs &args,
    const void *inptr, size_t ld_in_row, size_t ld_in_col, size_t ld_input_channel,
    void *outptr, size_t ld_out_matrix, size_t ld_out_row,
    unsigned int thread_id, unsigned int n_threads
  ) const override
  {
    execute_internal(
      args,
      reinterpret_cast<const TIn *>(inptr), ld_in_row, ld_in_col, ld_input_channel,
      reinterpret_cast<TOut *>(outptr), ld_out_matrix, ld_out_row,
      thread_id, n_threads
    );
  }

  /* Utility method to get a transposed variant of a kernel, this transposed
   * version simply calls the original kernel with the input row and column
   * strides swapped.
   */
  static constexpr Kernel get_transposed_kernel(const Kernel &kernel)
  {
    return [kernel] (
      const unsigned int n_channels,
      const TIn *const inptr, const size_t ld_in_row, const size_t ld_in_col,
      TOut *const outptr, const size_t ld_out
    ) {
      kernel(n_channels, inptr, ld_in_col, ld_in_row, outptr, ld_out);
    };
  }
};

}  // namespace weight_transform
}  // namespace winograd
}  // namespace arm_conv
