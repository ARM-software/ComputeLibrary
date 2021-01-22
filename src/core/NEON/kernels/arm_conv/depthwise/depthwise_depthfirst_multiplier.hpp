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

namespace arm_conv {
namespace depthwise {

namespace common
{
  template <typename strategy, typename F>
  void depthwise_multiplier_execute(
    const F execute_tile,
    typename strategy::input_type pad_value,
    const DepthwiseArgs &args,
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
    const size_t param_stride,
    const unsigned int output_height,
    const unsigned int output_width,
    void *const _output,
    const size_t ld_output_col,
    const size_t ld_output_row,
    const size_t ld_output_batch,
    void *const _working_space,
    const unsigned int thread_id,
    const unsigned int n_threads
  )
  {
    using TInput = typename strategy::input_type;
    using TOutput = typename strategy::return_type;

    // Determine what portion of the work to do.
    const unsigned int n_rows_per_thread = arm_gemm::iceildiv(output_height, n_threads);
    const int start_out_height = std::min(thread_id * n_rows_per_thread, output_height);
    const int end_out_height = std::min(start_out_height + n_rows_per_thread, output_height);

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input);
    TOutput *const outptr = static_cast<TOutput *>(_output);

    // To simplify the kernel, we process padded or non-NCHW-ordered input into
    // a form which can be consumed by the kernel. This data is stored here and
    // passed into the kernel as an array of N pointers (one per row of the
    // input).
    TInput rearranged_input[strategy::input_rows][strategy::input_col_quads*(16 / sizeof(TInput))];
    const TInput *inptrs[strategy::input_rows];

    // Create an array for the output pointers
    TOutput * _outptr_array[strategy::output_rows * strategy::output_cols];
    TOutput **const outptr_array = _outptr_array;

    // Allocate portions of the working space
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space);
    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space);

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

        for (int start_out_j = 0; start_out_j < static_cast<int>(output_width);)
        {
          const int start_in_j = start_out_j * strategy::stride_cols - args.padding.left;
          const int pad_left = -std::min(0, start_in_j);

          const int end_out_j = start_out_j + strategy::output_cols;
          const int end_in_j = start_in_j + strategy::input_cols;

          const auto pad_right = static_cast<unsigned int>(-std::min(static_cast<int>(input_width) - end_in_j, 0));
          const unsigned int valid_output_cols = std::min(
            end_out_j - start_out_j,
            static_cast<int>(output_width) - start_out_j
          );

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

          const uint8_t *params = static_cast<const uint8_t *>(parameters);

          // Loop over the input channels
          for (unsigned int in_c = 0; in_c < input_channels; in_c++)
          {
            // Construct the input array - first fill with padding values and
            // then fill in correct values.
            for (unsigned int i = 0; i < strategy::input_rows; i++)
            {
              for (unsigned int j = 0;
                   j < (16 / sizeof(TInput)) * strategy::input_col_quads; j++)
              {
                rearranged_input[i][j] = pad_value;
              }
              inptrs[i] = rearranged_input[i];
            }

            auto inptr_row = inptr_batch + in_c +
                             (start_in_i + pad_top) * ld_input_row +
                             (start_in_j + pad_left) * ld_input_col;
            if (ld_input_col == 1 && !pad_left &&
                start_in_j + (16 / sizeof(TInput)) * strategy::input_col_quads < input_width)
            {
              // The input tensor is already in NCHW format, and we're reading
              // an unpadded section of it - allow the kernel to read it
              // directly.
              for (unsigned int i = pad_top; i < strategy::input_rows - pad_bottom; i++)
              {
                inptrs[i] = inptr_row;
                inptr_row += ld_input_row;
              }
            }
            else
            {
              // Either the input tensor isn't in NCHW format, or we're reading
              // a padded section. Copy the relevant portion of the input here
              // and allow the kernel to read this.
              for (unsigned int i = pad_top; i < strategy::input_rows - pad_bottom; i++)
              {
                auto inptr_col = inptr_row;
                for (unsigned int j = pad_left; j < strategy::input_cols - pad_right; j++)
                {
                  rearranged_input[i][j] = *inptr_col;
                  inptr_col += ld_input_col;
                }
                inptr_row += ld_input_row;
              }
            }

            execute_tile(inptrs, outptr_array, params);

            // Progress the output pointers
            TOutput **outptr_pos = outptr_array;
            for (auto i = 0u; i < strategy::output_rows * strategy::output_cols; i++)
            {
              outptr_pos[i] += args.channel_multiplier;
            }

            // Progress the pointer into the parameters
            params += param_stride;
          }
        }
      }
    }
  }
}

template <class strategy>
class DepthwiseDepthfirstWithMultiplier :
  public DepthwiseCommon<typename strategy::input_type,
                         typename strategy::weight_type,
                         typename strategy::return_type>
{
  using TInput = typename strategy::input_type;
  using TWeight = typename strategy::weight_type;
  using TOutput = typename strategy::return_type;
  using TAccum = typename strategy::bias_type;

  size_t sizeof_output_buffer(unsigned int n_channels) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TOutput>(strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(n_channels, vl);
    return sizeof(TOutput) * rounded_channels;
  }

  public:
  DepthwiseDepthfirstWithMultiplier(const DepthwiseArgs &args) : DepthwiseCommon<TInput, TWeight, TOutput>(args)
  {
  }

  DepthwiseDepthfirstWithMultiplier(DepthwiseDepthfirstWithMultiplier &) = delete;
  DepthwiseDepthfirstWithMultiplier &operator=(DepthwiseDepthfirstWithMultiplier &) = delete;

  size_t get_storage_size(void) const override
  {
    // TODO What if we insert extra padding? Biases are a different size to the inputs, ...
    const unsigned int vl = arm_gemm::utils::get_vector_length<TInput>(strategy::vl_type);
    const auto rounded_channels = this->m_args.input_channels * arm_gemm::roundup(this->m_args.channel_multiplier, vl);
    return (1 + this->m_args.kernel_rows * this->m_args.kernel_cols) * rounded_channels * sizeof(TWeight);
  }

  void pack_parameters(void *_buffer, const void *_biases, const void *_weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    // TODO What if the kernel needs a different packing function?

    // Cast the pointers
    float *buffer = static_cast<float *>(_buffer);
    const float *biases = static_cast<const float *>(_biases);
    const float *const weights = static_cast<const float *>(_weights);

    const unsigned int vl = arm_gemm::utils::get_vector_length<TInput>(strategy::vl_type);
    ld_weight_col = (ld_weight_col == 0) ? this->m_args.channel_multiplier * this->m_args.input_channels : ld_weight_col;
    ld_weight_row = (ld_weight_row == 0) ? this->m_args.kernel_cols * ld_weight_col : ld_weight_row;

    for (unsigned int in_c = 0; in_c < this->m_args.input_channels; in_c++)
    {
      for (unsigned int n = 0; n < this->m_args.channel_multiplier; n += vl)
      {
        const unsigned int out_c = in_c * this->m_args.channel_multiplier + n;
        const unsigned int todo = std::min(vl, this->m_args.channel_multiplier - n);

        // Copy across the correct amount of bias (or 0)
        for (unsigned int i = 0; i < todo; i++)
        {
          buffer[i] = (biases == nullptr) ? 0 : biases[out_c + i];
        }
        buffer += vl;

        // Copy each of the weights in turn
        auto weights_row = weights + out_c;
        for (unsigned int i = 0; i < this->m_args.kernel_rows; i++)
        {
          auto weights_col = weights_row;

          for (unsigned int j = 0; j < this->m_args.kernel_cols; j++)
          {
            for (unsigned int m = 0; m < todo; m++)
            {
              buffer[m] = weights_col[m];
            }
            buffer += vl;

            weights_col += ld_weight_col;
          }

          weights_row += ld_weight_row;
        }
      }
    }
  }

  size_t get_working_size(const unsigned int n_threads, const unsigned int n_channels) const override
  {
    const unsigned int n_output_channels = n_channels * this->m_args.channel_multiplier;
    return n_threads * sizeof_output_buffer(n_output_channels);
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
    TAccum activation_min = std::numeric_limits<TAccum>::has_infinity ? -std::numeric_limits<TAccum>::infinity() : std::numeric_limits<TAccum>::min();
    TAccum activation_max = std::numeric_limits<TAccum>::has_infinity ? std::numeric_limits<TAccum>::infinity() : std::numeric_limits<TAccum>::max();

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

    // Need a stride over blocks of parameters
    const unsigned int vl = arm_gemm::utils::get_vector_length<TOutput>(strategy::vl_type);
    const unsigned int param_stride =
      arm_gemm::roundup(this->m_args.channel_multiplier, vl) *
      (sizeof(TAccum) + sizeof(TWeight) * strategy::kernel_rows * strategy::kernel_cols);

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input);
    TOutput *const outptr = static_cast<TOutput *>(_output);

    // To simplify the kernel, we process padded or non-NCHW-ordered input into
    // a form which can be consumed by the kernel. This data is stored here and
    // passed into the kernel as an array of N pointers (one per row of the
    // input).
    TInput rearranged_input[strategy::input_rows][strategy::input_col_quads*4];
    const TInput *inptrs[strategy::input_rows];

    // Create an array for the output pointers
    TOutput * _outptr_array[strategy::output_rows * strategy::output_cols];
    TOutput **const outptr_array = _outptr_array;

    // Allocate portions of the working space
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space) + get_working_size(thread_id, input_channels);
    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space);

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

        for (int start_out_j = 0; start_out_j < static_cast<int>(output_width);)
        {
          const int start_in_j = start_out_j * strategy::stride_cols - this->m_args.padding.left;
          const int pad_left = -std::min(0, start_in_j);

          const int end_out_j = start_out_j + strategy::output_cols;
          const int end_in_j = start_in_j + strategy::input_cols;

          const auto pad_right = static_cast<unsigned int>(-std::min(static_cast<int>(input_width) - end_in_j, 0));
          const unsigned int valid_output_cols = std::min(
            end_out_j - start_out_j,
            static_cast<int>(output_width) - start_out_j
          );

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

          const uint8_t *params = static_cast<const uint8_t *>(parameters);

          // Loop over the input channels
          for (unsigned int in_c = 0; in_c < input_channels; in_c++)
          {
            // Construct the input array - first fill with padding values and
            // then fill in correct values.
            for (unsigned int i = 0; i < strategy::input_rows; i++)
            {
              for (unsigned int j = 0; j < 4 * strategy::input_col_quads; j++)
              {
                rearranged_input[i][j] = static_cast<TInput>(0);
              }
              inptrs[i] = rearranged_input[i];
            }

            auto inptr_row = inptr_batch + in_c +
                             (start_in_i + pad_top) * ld_input_row +
                             (start_in_j + pad_left) * ld_input_col;
            if (ld_input_col == 1 && !pad_left &&
                start_in_j + 4 * strategy::input_col_quads < input_width)
            {
              // The input tensor is already in NCHW format, and we're reading
              // an unpadded section of it - allow the kernel to read it
              // directly.
              for (unsigned int i = pad_top; i < strategy::input_rows - pad_bottom; i++)
              {
                inptrs[i] = inptr_row;
                inptr_row += ld_input_row;
              }
            }
            else
            {
              // Either the input tensor isn't in NCHW format, or we're reading
              // a padded section. Copy the relevant portion of the input here
              // and allow the kernel to read this.
              for (unsigned int i = pad_top; i < strategy::input_rows - pad_bottom; i++)
              {
                auto inptr_col = inptr_row;
                for (unsigned int j = pad_left; j < strategy::input_cols - pad_right; j++)
                {
                  rearranged_input[i][j] = *inptr_col;
                  inptr_col += ld_input_col;
                }
                inptr_row += ld_input_row;
              }
            }

            {
#ifdef CYCLE_PROFILING
              auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(strategy::output_rows * strategy::output_cols * this->m_args.channel_multiplier * strategy::kernel_rows * strategy::kernel_cols));
#endif
              strat.kernel(
                inptrs, outptr_array, params,
                this->m_args.channel_multiplier,
                activation_min, activation_max
              );
            }

            // Progress the output pointers
            TOutput **outptr_pos = outptr_array;
            for (auto i = 0u; i < strategy::output_rows * strategy::output_cols; i++)
            {
              outptr_pos[i] += this->m_args.channel_multiplier;
            }

            // Progress the pointer into the parameters
            params += param_stride;
          }
        }
      }
    }
  }
};

}  // namespace depthwise
}  // namespace arm_conv
