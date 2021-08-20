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

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

#include <limits>

namespace arm_conv {
namespace depthwise {

template <class strategy>
class DepthwiseDepthfirst : public DepthwiseCommon<typename strategy::input_type,
                                                   typename strategy::weight_type,
                                                   typename strategy::return_type>
{
  using TInput = typename strategy::input_type;
  using TWeight = typename strategy::weight_type;
  using TOutput = typename strategy::return_type;
  using TAccum = typename strategy::bias_type;

  size_t sizeof_input_buffer(unsigned int n_input_channels) const
  {
    return sizeof(TInput) * n_input_channels;
  }

  size_t sizeof_output_buffer(unsigned int n_output_channels) const
  {
    return sizeof(TOutput) * n_output_channels;
  }

  public:

  DepthwiseDepthfirst(const DepthwiseArgs &args) : DepthwiseCommon<TInput, TWeight, TOutput>(args)
  {
  }

  DepthwiseDepthfirst(DepthwiseDepthfirst &) = delete;
  DepthwiseDepthfirst &operator=(DepthwiseDepthfirst &) = delete;

  size_t get_storage_size(void) const override
  {
    // TODO What if we insert extra padding? Biases are a different size to the inputs, ...
    const unsigned int vl = arm_gemm::utils::get_vector_length<TInput>(strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(this->m_args.input_channels, vl);
    return (1 + this->m_args.kernel_rows * this->m_args.kernel_cols) * rounded_channels * sizeof(TWeight);
  }

  void pack_parameters(void *_buffer, const void *_biases, const void *_weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    // TODO What if the kernel needs a different packing function?

    // Cast the pointers
    uint8_t *buffer = static_cast<uint8_t *>(_buffer);
    const TAccum *biases = static_cast<const TAccum *>(_biases);
    const TWeight *const weights = static_cast<const TWeight *>(_weights);

    const unsigned int vl = arm_gemm::utils::get_vector_length<TAccum>(strategy::vl_type);
    ld_weight_col = (ld_weight_col == 0) ? this->m_args.input_channels : ld_weight_col;
    ld_weight_row = (ld_weight_row == 0) ? this->m_args.kernel_cols * ld_weight_col : ld_weight_row;

    for (unsigned int n = 0; n < this->m_args.input_channels; n += vl)
    {
      const unsigned int todo = std::min(vl, this->m_args.input_channels - n);

      // Copy across the correct amount of bias (or 0)
      for (unsigned int i = 0; i < todo; i++)
      {
        reinterpret_cast<TAccum *>(buffer)[i] = (biases == nullptr) ? 0 : biases[n + i];
      }
      buffer += vl * sizeof(TAccum);

      // Copy each of the weights in turn
      auto weights_row = weights + n;
      for (unsigned int i = 0; i < this->m_args.kernel_rows; i++)
      {
        auto weights_col = weights_row;

        for (unsigned int j = 0; j < this->m_args.kernel_cols; j++)
        {
          for (unsigned int m = 0; m < todo; m++)
          {
            reinterpret_cast<TWeight *>(buffer)[m] = weights_col[m];
          }
          buffer += vl * sizeof(TWeight);

          weights_col += ld_weight_col;
        }

        weights_row += ld_weight_row;
      }
    }
  }

  size_t get_working_size(const unsigned int n_threads, const unsigned int n_channels) const override
  {
    const unsigned int n_output_channels = n_channels * this->m_args.channel_multiplier;
    return n_threads * (sizeof_output_buffer(n_output_channels) + sizeof_input_buffer(n_channels));
  }

  using DepthwiseCommon<typename strategy::input_type, typename strategy::weight_type, typename strategy::return_type>::execute;
  void execute(
    const unsigned int batches,
    const unsigned int input_height,
    const unsigned int input_width,
    const unsigned int input_channels,
    const PaddingValues &padding,
    const void *const _input,
    const size_t ld_input_col,
    const size_t ld_input_row,
    const size_t ld_input_batch,
    const void *const parameters,
    const unsigned int output_height,
    const unsigned int output_width,
    void *const _output,
    const size_t ld_output_col,
    const size_t ld_output_row,
    const size_t ld_output_batch,
    void *const _working_space,
    const unsigned int thread_id,
    const unsigned int n_threads
  ) const override
  {
    strategy strat(this->m_args.cpu_info);
#ifdef CYCLE_PROFILING
    arm_gemm::profiler prof;
#endif

    // Compute activation values
    TAccum activation_min, activation_max;
    std::tie(activation_min, activation_max) = get_default_activation_values<TAccum>();

    switch (this->m_args.activation.type)
    {
      case arm_gemm::Activation::Type::BoundedReLU:
        activation_max = static_cast<TAccum>(this->m_args.activation.param1);
        // Fall through
      case arm_gemm::Activation::Type::ReLU:
        activation_min = static_cast<TAccum>(0);
        break;
      default:
        break;
    }

    // Determine what portion of the work to do.
    const unsigned int n_rows_per_thread = arm_gemm::iceildiv(output_height, n_threads);
    const int start_out_height = std::min(thread_id * n_rows_per_thread, output_height);
    const int end_out_height = std::min(start_out_height + n_rows_per_thread, output_height);

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input);
    TOutput *const outptr = static_cast<TOutput *>(_output);

    // Create an array for the input pointers
    const TInput * _inptr_array[strategy::input_rows * strategy::input_cols];
    const TInput **const inptr_array = _inptr_array;

    // Create an array for the output pointers
    TOutput * _outptr_array[strategy::output_rows * strategy::output_cols];
    TOutput **const outptr_array = _outptr_array;

    // Allocate portions of the working space
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space) + get_working_size(thread_id, input_channels);
    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space);
    TInput *const input_buffer = reinterpret_cast<TInput *>(working_space + sizeof_output_buffer(input_channels * this->m_args.channel_multiplier));

    // Initialise the input buffer
    for (unsigned int c = 0; c < input_channels; c++)
    {
      input_buffer[c] = static_cast<TInput>(0);
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
           start_out_i += static_cast<int>(strategy::output_rows))
      {
        const int end_out_i = start_out_i + strategy::output_rows;
        const int start_in_i = start_out_i * strategy::stride_rows - padding.top;
        const int end_in_i = start_in_i + strategy::input_rows;

        // Compute top/bottom padding
        const auto pad_top = static_cast<unsigned int>(-std::min(start_in_i, 0));
        const auto pad_bottom = static_cast<unsigned int>(-std::min(static_cast<int>(input_height) - end_in_i, 0));
        const unsigned int valid_output_rows = std::min(
          end_out_i - start_out_i,
          static_cast<int>(output_height) - start_out_i
        );

        // Fill the input pointer array with padding values
        for (auto index = 0u; index < strategy::input_rows * strategy::input_cols; index++)
        {
          inptr_array[index] = input_buffer;
        }

        for (int start_out_j = 0; start_out_j < static_cast<int>(output_width);)
        {
          const int start_in_j = start_out_j * strategy::stride_cols - this->m_args.padding.left;
          const int pad_left = -std::min(0, start_in_j);

          // Compute how many output tiles we can compute with the direct kernel.
          int n_direct_tiles = 0;
          if (!pad_top && !pad_bottom && !pad_left)
          {
            // Determine the maximum number of tiles we could handle.
            n_direct_tiles = (output_width - start_out_j) / strategy::output_cols;

            // Continue to reduce this number as required to avoid reading
            // padding on the right edge.
            int end_in_j = start_in_j + n_direct_tiles * strategy::input_cols;
            int pad_right = std::max(0, end_in_j - static_cast<int>(input_width));

            while (pad_right && n_direct_tiles)
            {
              n_direct_tiles--;
              end_in_j -= strategy::input_cols;
              pad_right = std::max(0, end_in_j - static_cast<int>(input_width));
            }
          }

          // Use the unpadded kernel if we can, otherwise use the padded one.
          if (n_direct_tiles)
          {
            auto inptr = inptr_batch + start_in_i*ld_input_row + start_in_j*ld_input_col;
            auto outptr = outptr_batch + start_out_i*ld_output_row + start_out_j*ld_output_col;
            start_out_j += n_direct_tiles*strategy::output_cols;

#ifdef CYCLE_PROFILING
            auto p = prof.ScopedProfiler(PROFILE_KERNEL, 0);
#endif
            strat.direct_kernel(1, n_direct_tiles,
                                inptr, ld_input_row, ld_input_col,
                                outptr, ld_output_row, ld_output_col,
                                parameters, this->m_args.input_channels,
                                activation_min, activation_max);
            continue;
          }

          const int end_out_j = start_out_j + strategy::output_cols;
          const int end_in_j = start_in_j + strategy::input_cols;

          const auto pad_right = static_cast<unsigned int>(-std::min(static_cast<int>(input_width) - end_in_j, 0));
          const unsigned int valid_output_cols = std::min(
            end_out_j - start_out_j,
            static_cast<int>(output_width) - start_out_j
          );

          // Construct the input pointer array - fill the array with pointers to
          // the input buffer and then fill in the required values.
          for (auto i = pad_top; i < strategy::input_rows - pad_bottom; i++)
          {
            // Can skip over the left padding because we will have either the
            // same or less than the previous tile.
            unsigned int j = pad_left;
            const TInput *colptr = inptr_batch + (start_in_i + i) * ld_input_row + (start_in_j + j) * ld_input_col;
            const TInput **ptrs = inptr_array + i * strategy::input_cols + j;
            for (; j < strategy::input_cols - pad_right; j++)
            {
              *(ptrs++) = colptr;
              colptr += ld_input_col;
            }
            for (; j < strategy::input_cols; j++)
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
            for (; j < strategy::output_cols; j++)
            {
              *(outptr_pos++) = output_buffer;
            }
          }
          for (auto i = valid_output_rows; i < strategy::output_rows; i++)
          {
            for (auto j = 0u; j < strategy::output_cols; j++)
            {
              *(outptr_pos++) = output_buffer;
            }
          }

          start_out_j += strategy::output_cols;

#ifdef CYCLE_PROFILING
          // TODO Work number
          auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(0));
#endif
          strat.indirect_kernel(inptr_array, outptr_array, parameters,
                                this->m_args.input_channels, activation_min, activation_max);
        }
      }
    }
  }
};

}  // namespace depthwise
}  // namespace arm_conv
