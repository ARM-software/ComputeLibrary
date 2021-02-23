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

#include "arm_compute/core/Types.h"
#include <limits>

namespace arm_conv {
namespace pooling {

template <class strategy>
class PoolingDepthfirst : public PoolingCommon<typename strategy::operand_type, typename strategy::return_type>
{
  using TInput = typename strategy::operand_type;
  using TOutput = typename strategy::return_type;

  const PoolingArgs m_args;  // Copy of arguments

  constexpr static unsigned int input_rows(void)
  {
    return (strategy::out_rows() - 1)*strategy::stride_rows() + strategy::pool_rows();
  }

  constexpr static unsigned int input_cols(void)
  {
    return (strategy::out_cols() - 1)*strategy::stride_cols() + strategy::pool_cols();
  }

  size_t sizeof_input_buffer(void) const
  {
    return sizeof(TInput) * m_args.n_channels;
  }

  size_t sizeof_output_buffer(void) const
  {
    return sizeof(TOutput) * m_args.n_channels;
  }

  public:
  PoolingDepthfirst(const PoolingArgs &args) : m_args(args)
  {
  }

  PoolingDepthfirst(PoolingDepthfirst &) = delete;
  PoolingDepthfirst &operator=(PoolingDepthfirst &) = delete;

  size_t get_working_size(unsigned int num_threads) const override
  {
    // We require a channel-length vector of input padding values
    // (to be shared amongst all threads) and (for each thread) a
    // channel-length vector in which to dump surplus output.
    return sizeof_input_buffer() + num_threads * sizeof_output_buffer();
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
    ARM_COMPUTE_UNUSED(batches, ld_input_batch, ld_output_batch);
    strategy strat(m_args.cpu_info);
#ifdef CYCLE_PROFILING
    arm_gemm::profiler prof;
#endif // CYCLE_PROFILING

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input);
    TOutput *const outptr = static_cast<TOutput *>(_output);

    const unsigned int roundup_output_rows = roundup(output_height, num_threads);
    const unsigned int rows_per_thread = roundup_output_rows / num_threads;
    const int start_out_height = static_cast<int>(thread_id * rows_per_thread);
    const int end_out_height = std::min<int>(output_height, static_cast<int>((thread_id + 1) * rows_per_thread));

    // Create an array for the input pointers
    const TInput * _inptr_array[input_rows() * input_cols()];
    const TInput **const inptr_array = _inptr_array;

    // Create an array for the output pointers
    TOutput * _outptr_array[strategy::out_rows() * strategy::out_cols()];
    TOutput **const outptr_array = _outptr_array;

    // Allocate portions of the working space
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space);
    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space + thread_id * sizeof_output_buffer());
    TInput *const input_buffer = reinterpret_cast<TInput *>(working_space + num_threads * sizeof_output_buffer());

    // Initialise the input buffer
    for (unsigned int c = 0; c < channels; c++)
    {
      TInput &val = input_buffer[c];

      if (strategy::pooling_type() == PoolingType::AVERAGE)
      {
        val = static_cast<TInput>(0);
      }
      else if (strategy::pooling_type() == PoolingType::MAX)
      {
#if defined(__aarch64__)
        using InputType = typename std::conditional<std::is_same<TInput, __fp16>::value, arm_compute::half, TInput>::type;
        using limits = std::numeric_limits<InputType>;
#else // defined(__aarch64__)
        using limits = std::numeric_limits<TInput>;
#endif // defined(__aarch64__)
        if (limits::has_infinity)
        {
          val = -limits::infinity();
        }
        else
        {
          val = limits::min();
        }
      }
    }

    // For each output tile, construct the requisite set of pointers and call
    // into the kernel.
    for (unsigned int batch = 0; batch < batches; batch++)
    {
      // Get batch pointers
      const auto inptr_batch = inptr + batch * ld_input_batch;
      const auto outptr_batch = outptr + batch * ld_output_batch;

      for (int start_out_i = start_out_height;
           start_out_i < end_out_height;
           start_out_i += static_cast<int>(strategy::out_rows()))
      {
        const int end_out_i = start_out_i + strategy::out_rows();
        const int start_in_i = start_out_i * strategy::stride_rows() - padding.top;
        const int end_in_i = start_in_i + input_rows();

        // Compute top/bottom padding - TODO Is this right for average pooling?
        const auto pad_top = static_cast<unsigned int>(-std::min(start_in_i, 0));
        const auto pad_bottom = static_cast<unsigned int>(-std::min(static_cast<int>(height) - end_in_i, 0));
        const unsigned int valid_output_rows = std::min(
          end_out_i - start_out_i,
          static_cast<int>(end_out_height) - start_out_i
        );

        // Fill the input pointer array with padding values
        for (auto index = 0u; index < input_rows() * input_cols(); index++)
        {
          inptr_array[index] = input_buffer;
        }

        for (int start_out_j = 0, start_in_j = -padding.left;
             start_out_j < static_cast<int>(output_width);
             start_out_j += static_cast<int>(strategy::out_cols()),
             start_in_j += static_cast<int>(strategy::out_cols()) * strategy::stride_cols())
        {
          const int end_out_j = start_out_j + strategy::out_cols();
          const int end_in_j = start_in_j + input_cols();

          // Compute left/right padding - TODO Is this right for average pooling?
          const auto pad_left = static_cast<unsigned int>(-std::min(start_in_j, 0));
          const auto pad_right = static_cast<unsigned int>(-std::min(static_cast<int>(width) - end_in_j, 0));

          const unsigned int valid_output_cols = std::min(
            end_out_j - start_out_j,
            static_cast<int>(output_width) - start_out_j
          );

          // Construct the input pointer array - fill the array with pointers to
          // the input buffer and then fill in the required values.
          for (auto i = pad_top; i < input_rows() - pad_bottom; i++)
          {
            // Can skip over the left padding because we will have either the
            // same or less than the previous tile.
            unsigned int j = pad_left;
            const TInput *colptr = inptr_batch + (start_in_i + i) * ld_input_row + (start_in_j + j) * ld_input_col;
            const TInput **ptrs = inptr_array + i * input_cols() + j;
            for (; j < input_cols() - pad_right; j++)
            {
              *(ptrs++) = colptr;
              colptr += ld_input_col;
            }
            for (; j < input_cols(); j++)
            {
              *(ptrs++) = input_buffer;
            }
          }

          // Construct the output pointer array.
          TOutput **outptr_pos = outptr_array;
          for (auto i = 0u; i < valid_output_rows; i++)
          {
            unsigned int j = 0u;
            TOutput *colptr = outptr_batch + (start_out_i + i) * ld_output_row + start_out_j * ld_output_col;
            for (; j < valid_output_cols; j++)
            {
              *(outptr_pos++) = colptr;
               colptr += ld_output_col;
            }
            for (; j < strategy::out_cols(); j++)
            {
              *(outptr_pos++) = output_buffer;
            }
          }
          for (auto i = valid_output_rows; i < strategy::out_rows(); i++)
          {
            for (auto j = 0u; j < strategy::out_cols(); j++)
            {
              *(outptr_pos++) = output_buffer;
            }
          }

#ifdef CYCLE_PROFILING
          // TODO Work number
          auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(strategy::out_rows() * strategy::out_cols() * strategy::pool_rows() * strategy::pool_cols()));
#endif
          strat.kernel(
            channels, inptr_array, outptr_array,
            m_args.exclude_padding, pad_left, pad_top, pad_right, pad_bottom
          );
        }
      }
    }
  }
};

}  // namespace pooling
}  // namespace arm_conv
