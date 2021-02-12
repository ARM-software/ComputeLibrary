/*
 * Copyright (c) 2021 Arm Limited.
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

#include "pool_common.hpp"
#include "utils.hpp"

namespace arm_conv {
namespace pooling {

template <class strategy>
class PoolingDepthfirstGenericQuantized : public PoolingCommon<typename strategy::operand_type, typename strategy::return_type, Requantize32>
{
  using TInput = typename strategy::operand_type;
  using TOutput = typename strategy::return_type;

  const PoolingArgs m_args;  // Copy of arguments
  const Requantize32 m_requant;  // Quantization parameters

  unsigned int input_rows(void) const
  {
    return m_args.pool_window.rows;
  }

  unsigned int input_cols(void) const
  {
    return m_args.pool_window.cols;
  }

  public:
  PoolingDepthfirstGenericQuantized(const PoolingArgs &args, const Requantize32 &rq) : m_args(args), m_requant(rq)
  {
  }

  PoolingDepthfirstGenericQuantized(PoolingDepthfirstGenericQuantized &) = delete;
  PoolingDepthfirstGenericQuantized &operator=(PoolingDepthfirstGenericQuantized &) = delete;

  size_t sizeof_input_pointer_array(void) const
  {
    return sizeof(TInput *) * input_rows() * input_cols();
  }

  size_t get_working_size(unsigned int num_threads) const override
  {
    return num_threads * sizeof_input_pointer_array();
  }

  void execute(
    const void *const input,
    void *const output,
    void *const working_space,
    unsigned int thread_id,
    unsigned int num_threads
  ) const override
  {
    const size_t ld_input_col = m_args.n_channels;
    const size_t ld_input_row = ld_input_col * m_args.input_cols;
    const size_t ld_input_batch = ld_input_row * m_args.input_rows;
    const size_t ld_output_col = ld_input_col;
    const size_t ld_output_row = ld_output_col * m_args.output_cols;
    const size_t ld_output_batch = ld_output_row * m_args.output_rows;

    execute(
      input, ld_input_col, ld_input_row, ld_input_batch,
      output, ld_output_col, ld_output_row, ld_output_batch,
      working_space,
      thread_id, num_threads
    );
  }

  void execute(
    const void *const input,
    size_t ld_input_col,
    size_t ld_input_row,
    size_t ld_input_batch,
    void *const output,
    size_t ld_output_col,
    size_t ld_output_row,
    size_t ld_output_batch,
    void *const working_space,
    unsigned int thread_id,
    unsigned int num_threads
  ) const override
  {
    execute(
      m_args.n_batches, m_args.input_rows, m_args.input_cols,
      m_args.n_channels,
      input, ld_input_col, ld_input_row, ld_input_batch,
      m_args.padding,
      m_args.output_rows, m_args.output_cols,
      output, ld_output_col, ld_output_row, ld_output_batch,
      working_space,
      thread_id, num_threads
    );
  }

  void execute(
    unsigned int batches,
    unsigned int height,
    unsigned int width,
    unsigned int channels,
    const void *const _input,
    size_t ld_input_col,
    size_t ld_input_row,
    size_t ld_input_batch,
    const PaddingValues &padding,
    unsigned int output_height,
    unsigned int output_width,
    void *const _output,
    size_t ld_output_col,
    size_t ld_output_row,
    size_t ld_output_batch,
    void *const _working_space,
    unsigned int thread_id,
    unsigned int num_threads
  ) const override
  {
    strategy strat(m_args.cpu_info);
#ifdef CYCLE_PROFILING
    arm_gemm::profiler prof;
#endif // CYCLE_PROFILING

    const unsigned int roundup_output_rows = roundup(output_height, num_threads);
    const unsigned int rows_per_thread = roundup_output_rows / num_threads;
    int start_out_height = static_cast<int>(thread_id * rows_per_thread);
    int end_out_height = std::min<int>(output_height, static_cast<int>((thread_id + 1) * rows_per_thread));

    unsigned int start_channel = 0;
    unsigned int end_channel = channels;
    if(output_height == 1)
    {
      const unsigned int channels_per_thread = roundup(channels, num_threads) / num_threads;
      start_channel = thread_id * channels_per_thread;
      end_channel = std::min(start_channel + channels_per_thread, channels);

      // Reset start and end rows
      start_out_height = 0;
      end_out_height = output_height;
    }

    if(start_channel >= end_channel)
    {
        // Early exit in case of multiple threads parallelising on channels
        return;
    }

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input) + start_channel;
    TOutput *const outptr = static_cast<TOutput *>(_output) + start_channel;

    // Grab the input pointer array
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space);
    const TInput **const inptr_array = reinterpret_cast<const TInput **>(working_space + thread_id * sizeof_input_pointer_array());

    // For each output tile, construct the requisite set of pointers and call
    // into the kernel.
    for (unsigned int batch = 0; batch < batches; batch++)
    {
      // Get batch pointers
      const auto inptr_batch = inptr + batch * ld_input_batch;
      const auto outptr_batch = outptr + batch * ld_output_batch;

      for (int out_i = start_out_height; out_i < end_out_height; out_i++)
      {
        const int start_in_i = out_i * m_args.pool_stride.rows - padding.top;
        const int end_in_i = start_in_i + m_args.pool_window.rows;

        // Compute top/bottom padding
        const auto pad_top = static_cast<unsigned int>(-std::min(start_in_i, 0));
        const auto pad_bottom = static_cast<unsigned int>(-std::min(static_cast<int>(height) - end_in_i, 0));

        // Compute the number of pooling window rows which are contained in
        // either the valid region of the input tensor, or the padding.
        const auto padded_bottom = std::min<unsigned int>(
          start_in_i + m_args.pool_window.rows, height + padding.bottom
        );
        const auto n_total_rows = padded_bottom - start_in_i;

        for (int out_j = 0, start_in_j = -padding.left;
             out_j < static_cast<int>(output_width);
             out_j++, start_in_j += m_args.pool_stride.cols)
        {
          const int end_in_j = start_in_j + m_args.pool_window.cols;

          // Compute left/right padding
          const auto pad_left = static_cast<unsigned int>(-std::min(start_in_j, 0));
          const auto pad_right = static_cast<unsigned int>(-std::min(static_cast<int>(width) - end_in_j, 0));

          // Compute the number of pooling window columns which are contained
          // in either the valid region of the input tensor, or the padding.
          const auto padded_right = std::min<unsigned int>(
            start_in_j + m_args.pool_window.cols, width + padding.right
          );
          const auto n_total_cols = padded_right - start_in_j;

          // Construct the input pointer array - fill in all valid points
          // contiguously.
          const TInput **ptrs = inptr_array;
          for (auto i = pad_top; i < input_rows() - pad_bottom; i++)
          {
            // Can skip over the left padding because we will have either the
            // same or less than the previous tile.
            unsigned int j = pad_left;
            const TInput *colptr = inptr_batch + (start_in_i + i) * ld_input_row + (start_in_j + j) * ld_input_col;
            for (; j < input_cols() - pad_right; j++)
            {
              *(ptrs++) = colptr;
              colptr += ld_input_col;
            }
          }

          // Compute the number of valid cells
          const auto valid_rows = input_rows() - pad_top - pad_bottom;
          const auto valid_cols = input_cols() - pad_left - pad_right;
          const auto valid_cells = valid_rows * valid_cols;
          const auto cells_in_range = n_total_rows * n_total_cols;
          const auto window_cells = m_args.exclude_padding ? valid_cells : cells_in_range;

          // Get the output pointer for this call
          TOutput *outptr = outptr_batch + out_i * ld_output_row + out_j * ld_output_col;

#ifdef CYCLE_PROFILING
          // TODO Work number
          auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long) 0);
#endif
          strat.kernel(window_cells, valid_cells, end_channel - start_channel, inptr_array, outptr, m_requant);
        }
      }
    }
  }
};

}  // namespace pooling
}  // namespace arm_conv
