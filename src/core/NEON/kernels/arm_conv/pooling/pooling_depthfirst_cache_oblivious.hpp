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

#include <stack>
#include <vector>

namespace arm_conv {
namespace pooling {

template <class strategy>
class PoolingDepthfirstCacheOblivious : public PoolingCommon<typename strategy::operand_type, typename strategy::return_type>
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
  PoolingDepthfirstCacheOblivious(const PoolingArgs &args) : m_args(args)
  {
  }

  PoolingDepthfirstCacheOblivious(PoolingDepthfirstCacheOblivious &) = delete;
  PoolingDepthfirstCacheOblivious &operator=(PoolingDepthfirstCacheOblivious &) = delete;

  size_t get_working_size(void) const override
  {
    // We require an array of pointers for the inputs and outputs, a
    // channel-length vector in which to dump surplus output, and a
    // channel-length vector of padding values.
    return sizeof_input_buffer() + sizeof_output_buffer();
  }

  void execute(
    const void *const input,
    void *const output,
    void *const working_space
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
      working_space
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
    void *const working_space
  ) const override
  {
    execute(
      m_args.n_batches, m_args.input_rows, m_args.input_cols,
      m_args.n_channels,
      input, ld_input_col, ld_input_row, ld_input_batch,
      m_args.padding,
      m_args.output_rows, m_args.output_cols,
      output, ld_output_col, ld_output_row, ld_output_batch,
      working_space
    );
  }

  void execute(
    unsigned int batches,
    unsigned int input_height,
    unsigned int input_width,
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
    void *const _working_space
  ) const override
  {
    strategy strat(m_args.cpu_info);
#ifdef CYCLE_PROFILING
    arm_gemm::profiler prof;
#endif // CYCLE_PROFILING

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input);
    TOutput *const outptr = static_cast<TOutput *>(_output);

    // Allocate portions of the working space
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space);
    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space);
    TInput *const input_buffer = reinterpret_cast<TInput *>(working_space + sizeof_output_buffer());

    // Fill the input buffer
    const TInput pad_value = (m_args.pool_type == PoolingType::AVERAGE)
                           ? static_cast<TInput>(0)
                           : (std::numeric_limits<TInput>::has_infinity
                              ? -std::numeric_limits<TInput>::infinity()
                              : std::numeric_limits<TInput>::lowest());
    for (unsigned int i = 0; i < channels; i++)
    {
      input_buffer[i] = pad_value;
    }

    // Keep subdividing the output plane across the longest dimension until we
    // reach the size of the tile. Queue items for later processing. Note - we
    // can determine the largest size of the queue a priori from the input
    // tensor size, this would allow us to allocate memory within the working
    // space and improve performance.
    struct WorkItem
    {
      unsigned int output_i, output_j;
      unsigned int output_height, output_width;

      WorkItem(unsigned int i, unsigned int j, unsigned int height, unsigned int width)
        : output_i(i), output_j(j), output_height(height), output_width(width) {}
    };

    auto execute = [&] (const WorkItem &item) {
      // Create an array for the output pointers
      TOutput * _outptr_array[strategy::out_rows() * strategy::out_cols()];
      TOutput **const outptr_array = _outptr_array;

      // Construct the output pointer array
      {
        const auto output_pad_right = strategy::out_rows() - item.output_width;
        auto outptr_element = outptr_array;
        auto outptr_row = outptr + item.output_i * ld_output_row + item.output_j * ld_output_col;

        // Fill the array with pointers to the output buffer
        for (unsigned int i = 0; i < strategy::out_rows() * strategy::out_cols(); i++)
        {
          outptr_array[i] = output_buffer;
        }

        // Fill in the valid portion of the array
        for (unsigned int i = 0; i < item.output_height; i++)
        {
          auto outptr_col = outptr_row;
          for (unsigned int j = 0; j < item.output_width; j++)
          {
            *(outptr_element++) = outptr_col;
            outptr_col += ld_output_col;
          }
          outptr_element += output_pad_right;
          outptr_row += ld_output_row;
        }
      }

      const int start_i = item.output_i * strategy::stride_rows() - padding.top;
      const int end_i = start_i + input_rows();
      const unsigned int pad_top = std::max(0, 0 - start_i);
      const unsigned int pad_bottom = std::max(0, end_i - static_cast<int>(input_height));

      const int start_j = item.output_j * strategy::stride_cols() - padding.left;
      const int end_j = start_j + input_cols();
      const unsigned int pad_left = std::max(0, 0 - start_j);
      const unsigned int pad_right = std::max(0, end_j - static_cast<int>(input_width));

      // Create an array for the input pointers
      const TInput * _inptr_array[input_rows() * input_cols()];
      const TInput **const inptr_array = _inptr_array;
      {
        const unsigned int row_padding = pad_top + pad_bottom;
        const unsigned int valid_rows = input_rows() - row_padding;

        const unsigned int col_padding = pad_left + pad_right;
        const unsigned int valid_cols = input_cols() - col_padding;

        // Fill the array with pointers to the input buffer
        for (unsigned int i = 0; i < input_rows() * input_cols(); i++)
        {
          inptr_array[i] = input_buffer;
        }

        // Compute valid initial pointer
        auto inptr_row = inptr + std::max(start_i, 0) * ld_input_row + std::max(start_j, 0) * ld_input_col;

        // Fill in the valid portion of the input array
        auto inptr_element = inptr_array + pad_top * input_cols() + pad_left;
        for (unsigned int i = 0; i < valid_rows; i++)
        {
          auto inptr_col = inptr_row;
          for (unsigned int j = 0; j < valid_cols; j++)
          {
            *(inptr_element++) = inptr_col;
            inptr_col += ld_input_col;
          }

          inptr_row += ld_input_row;
          inptr_element += col_padding;  // Skip the padding elements
        }
      }

      // Call the kernel
#ifdef CYCLE_PROFILING
      // TODO Work number
      auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(item.output_height * item.output_width * strategy::pool_rows() * strategy::pool_cols()));
#endif // CYCLE_PROFILING
      strat.kernel(channels, inptr_array, outptr_array,
                   pad_left, pad_top, pad_right, pad_bottom);
    };

    // Add the initial work item to the stack of work.
    std::stack<WorkItem, std::vector<WorkItem>> stack;
    stack.push(WorkItem(0, 0, output_height, output_width));
    while (!stack.empty())
    {
      // Pop an item from the stack, bisect the largest dimension and either
      // execute the resulting tiles or add them to the stack if they are too
      // large.
      const WorkItem item(stack.top());
      stack.pop();

      if (item.output_height <= strategy::out_rows() &&
          item.output_width <= strategy::out_cols())
      {
        execute(item);
      }
      else
      {
        // Split the largest dimension, such that we get an exact number of
        // tiles in the first partition.
        if (item.output_height >= item.output_width)
        {
          const unsigned int height_in_tiles = (item.output_height + strategy::out_rows() - 1) / strategy::out_rows();
          const unsigned int tiles_first = height_in_tiles - height_in_tiles / 2;

          const unsigned int height_first = tiles_first * strategy::out_rows();
          const unsigned int height_second = item.output_height - height_first;

          stack.push(WorkItem(item.output_i + height_first, item.output_j, height_second, item.output_width));
          stack.push(WorkItem(item.output_i, item.output_j, height_first, item.output_width));
        }
        else
        {
          const unsigned int width_in_tiles = item.output_width / strategy::out_cols();
          const unsigned int tiles_first = width_in_tiles - width_in_tiles / 2;

          const unsigned int width_first = tiles_first * strategy::out_cols();
          const unsigned int width_second = item.output_width - width_first;

          stack.push(WorkItem(item.output_i, item.output_j + width_first, item.output_height, width_second));
          stack.push(WorkItem(item.output_i, item.output_j, item.output_height, width_first));
        }
      }
    }
  }
};

}  // namespace pooling
}  // namespace arm_conv
