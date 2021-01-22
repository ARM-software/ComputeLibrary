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

template <class strategy>
class DepthwiseDepthfirstGenericWithMultiplierBase :
  public DepthwiseCommon<typename strategy::input_type,
                         typename strategy::weight_type,
                         typename strategy::return_type>
{
  protected:

  using TInput = typename strategy::input_type;
  using TWeight = typename strategy::weight_type;
  using TOutput = typename strategy::return_type;
  using TAccum = typename strategy::bias_type;

  unsigned int kernel_points(void) const
  {
    return this->m_args.kernel_rows * this->m_args.kernel_cols;
  }

  unsigned int input_rows(void) const
  {
    return (strategy::output_rows() - 1) * this->m_args.stride_rows + this->m_args.kernel_rows;
  }

  unsigned int input_cols(void) const
  {
    return (strategy::output_cols() - 1) * this->m_args.stride_cols + this->m_args.kernel_cols;
  }

  size_t sizeof_inptr_array(void) const
  {
    return sizeof(TInput *) * kernel_points() * strategy::output_rows();
  }

  size_t sizeof_input_samples(void) const
  {
    // We have a sample for each kernel point, for each point of the output array.
    return sizeof(TInput) * kernel_points() *
                            strategy::output_rows() *
                            strategy::output_col_regs() *
                            (16 / sizeof(TAccum));
  }

  size_t sizeof_outptr_array(void) const
  {
    return sizeof(TOutput *) * strategy::output_rows() * strategy::output_cols();
  }

  size_t sizeof_output_buffer(unsigned int n_channels) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TOutput>(strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(n_channels, vl);
    return sizeof(TOutput) * rounded_channels;
  }

  void pack_weights(TWeight *buffer, const TWeight *weights, size_t ld_weight_col, size_t ld_weight_row) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TAccum>(strategy::vl_type);
    ld_weight_col = (ld_weight_col == 0) ? this->m_args.channel_multiplier * this->m_args.input_channels : ld_weight_col;
    ld_weight_row = (ld_weight_row == 0) ? this->m_args.kernel_cols * ld_weight_col : ld_weight_row;

    for (unsigned int in_c = 0; in_c < this->m_args.input_channels; in_c++)
    {
      for (unsigned int n = 0; n < this->m_args.channel_multiplier; n += vl)
      {
        const unsigned int out_c = in_c * this->m_args.channel_multiplier + n;
        const unsigned int todo = std::min(vl, this->m_args.channel_multiplier - n);

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

  void execute_tiles(
    std::function<void(const TInput **, TOutput **, const TWeight *, unsigned int, unsigned int)> tile_fn,
    const TInput pad_value,
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
  ) const
  {
#ifdef CYCLE_PROFILING
    arm_gemm::profiler prof;
#endif

    // Determine what portion of the work to do.
    const unsigned int n_rows_per_thread = arm_gemm::iceildiv(output_height, n_threads);
    const int start_out_height = std::min(thread_id * n_rows_per_thread, output_height);
    const int end_out_height = std::min(start_out_height + n_rows_per_thread, output_height);

    // Need a stride over blocks of parameters
    const unsigned int vl = arm_gemm::utils::get_vector_length<TAccum>(strategy::vl_type);
    const unsigned int param_stride = arm_gemm::roundup(this->m_args.channel_multiplier, vl) * kernel_points();

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input);
    TOutput *const outptr = static_cast<TOutput *>(_output);

    // Allocate portions of the working space
    uint8_t *working_space = static_cast<uint8_t *>(_working_space) +
                             get_working_size(thread_id, input_channels);

    const TInput **inptrs = reinterpret_cast<const TInput **>(working_space);
    working_space += sizeof_inptr_array();

    // To simplify the kernel, we process padded or non-NCHW-ordered input into
    // a form which can be consumed by the kernel. This data is stored here and
    // passed into the kernel as an array of N pointers (one per row of the
    // input).
    TInput *rearranged_input = reinterpret_cast<TInput *>(working_space);
    working_space += sizeof_input_samples();

    TOutput **outptr_array = reinterpret_cast<TOutput **>(working_space);
    working_space += sizeof_outptr_array();

    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space);

    // TODO Dynamically change the input pointer array in cases where we could
    // read directly from the input tensor; for now though assume we will
    // always read from the sample array.
    {
      auto my_inptrs = inptrs;
      auto my_input_samples = rearranged_input;

      // For each kernel point; for each row of output; for each register of
      // values containing a QUAD of source values.
      const unsigned int quad_length = 16 / sizeof(TAccum);

      for (auto p = 0u; p < kernel_points() * strategy::output_rows(); p++)
      {
        *(my_inptrs)++ = my_input_samples;
        my_input_samples += arm_gemm::roundup(strategy::output_cols(), quad_length);
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
           start_out_i += static_cast<int>(strategy::output_rows()))
      {
        const int end_out_i = std::min(start_out_i + static_cast<int>(strategy::output_rows()), end_out_height);
        const int start_in_i = start_out_i * this->m_args.stride_rows - padding.top;
        const int end_in_i = start_in_i + input_rows();

        // Compute top/bottom padding
        const auto pad_top = static_cast<unsigned int>(-std::min(start_in_i, 0));
        const auto pad_bottom = static_cast<unsigned int>(-std::min(static_cast<int>(input_height) - end_in_i, 0));
        const unsigned int valid_output_rows = std::min(
          end_out_i - start_out_i,
          static_cast<int>(output_height) - start_out_i
        );

        const int pad_rows = pad_top + pad_bottom;

        for (int start_out_j = 0; start_out_j < static_cast<int>(output_width);)
        {
          const int start_in_j = start_out_j * this->m_args.stride_cols - this->m_args.padding.left;
          const int pad_left = -std::min(0, start_in_j);

          const int end_out_j = start_out_j + strategy::output_cols();
          const int end_in_j = start_in_j + input_cols();

          const auto pad_right = static_cast<unsigned int>(-std::min(static_cast<int>(input_width) - end_in_j, 0));
          const unsigned int valid_output_cols = std::min(
            end_out_j - start_out_j,
            static_cast<int>(output_width) - start_out_j
          );

          const int pad_cols = pad_left + pad_right;

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
            for (; j < strategy::output_cols(); j++)
            {
              *(outptr_pos++) = output_buffer;
            }
          }
          for (auto i = valid_output_rows; i < strategy::output_rows(); i++)
          {
            for (auto j = 0u; j < strategy::output_cols(); j++)
            {
              *(outptr_pos++) = output_buffer;
            }
          }

          start_out_j += strategy::output_cols();

          const TWeight *params = static_cast<const TWeight *>(parameters);

          // Fill the input samples with padding. We can do this outside of
          // the channel loop, as the position of padding isn't going to
          // change as a function of channel.
          for (auto i = 0u; i < kernel_points() * strategy::output_rows() * strategy::output_cols(); i++)
          {
            rearranged_input[i] = pad_value;
          }

          // Loop over the input channels
          for (unsigned int in_c = 0; in_c < input_channels; in_c++)
          {
            auto inptr_row = inptr_batch + in_c +
                             (start_in_i + pad_top) * ld_input_row +
                             (start_in_j + pad_left) * ld_input_col;

            // Construct the array of input samples; for each point of the
            // kernel we provide an input value for each output point.
            auto input_samples = rearranged_input;
            for (auto ki = 0u; ki < this->m_args.kernel_rows; ki++)
            {
              for (auto kj = 0u; kj < this->m_args.kernel_cols; kj++)
              {
                // Copy the pointer for the input samples associated with this
                // kernel point. Then update the main pointer to account for
                // this point.
                auto point_input_samples = input_samples;
                input_samples += strategy::output_rows() * strategy::output_cols();

                int ii = static_cast<int>(ki) - static_cast<int>(pad_top);
                for (auto oi = 0u;
                     oi < strategy::output_rows() &&
                     ii < static_cast<int>(input_rows()) - pad_rows;
                     oi++, ii += this->m_args.stride_rows)
                {
                  if (0 <= ii) // Fill in values only if this row is in range.
                  {
                    int ij = static_cast<int>(kj) - static_cast<int>(pad_left);
                    for (auto oj = 0u;
                         oj < strategy::output_cols() &&
                         ij < static_cast<int>(input_cols()) - pad_cols;
                         oj++, ij += this->m_args.stride_cols)
                    {
                      if (0 <= ij) // Sample if the point is in range.
                      {
                        point_input_samples[oj] = *(inptr_row + ii*ld_input_row + ij*ld_input_col);
                      }
                    }
                  }

                  point_input_samples += strategy::output_cols();
                }
              }
            }

            tile_fn(inptrs, outptr_array, params, in_c, in_c*this->m_args.channel_multiplier);

            // Progress the output pointers
            TOutput **outptr_pos = outptr_array;
            for (auto i = 0u; i < strategy::output_rows() * strategy::output_cols(); i++)
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

  public:
  DepthwiseDepthfirstGenericWithMultiplierBase(const DepthwiseArgs &args) : DepthwiseCommon<TInput, TWeight, TOutput>(args)
  {
  }

  DepthwiseDepthfirstGenericWithMultiplierBase(DepthwiseDepthfirstGenericWithMultiplierBase &) = delete;
  DepthwiseDepthfirstGenericWithMultiplierBase &operator=(DepthwiseDepthfirstGenericWithMultiplierBase &) = delete;

  size_t get_storage_size(void) const override
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TAccum>(strategy::vl_type);
    const auto rounded_channels = this->m_args.input_channels * arm_gemm::roundup(this->m_args.channel_multiplier, vl);
    return kernel_points() * rounded_channels * sizeof(TWeight);
  }

  size_t get_working_size(const unsigned int n_threads, const unsigned int n_channels) const override
  {
    const unsigned int n_output_channels = n_channels * this->m_args.channel_multiplier;
    return n_threads * (sizeof_inptr_array() +
                        sizeof_input_samples() +
                        sizeof_outptr_array() +
                        sizeof_output_buffer(n_output_channels));
  }
};

template <class strategy>
class DepthwiseDepthfirstGenericWithMultiplier : public DepthwiseDepthfirstGenericWithMultiplierBase<strategy>
{
  using TInput = typename strategy::input_type;
  using TWeight = typename strategy::weight_type;
  using TOutput = typename strategy::return_type;
  using TAccum = typename strategy::bias_type;

  using Parent = DepthwiseDepthfirstGenericWithMultiplierBase<strategy>;

  const TAccum *m_biases;  // Pointer to bias vector

  public:
  DepthwiseDepthfirstGenericWithMultiplier(const DepthwiseArgs &args)
    : Parent(args), m_biases(nullptr)
  {
  }

  DepthwiseDepthfirstGenericWithMultiplier(DepthwiseDepthfirstGenericWithMultiplier &) = delete;
  DepthwiseDepthfirstGenericWithMultiplier &operator=(DepthwiseDepthfirstGenericWithMultiplier &) = delete;

  void pack_parameters(void *buffer, const void *biases, const void *weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    m_biases = static_cast<const TAccum *>(biases);
    Parent::pack_weights(static_cast<TAccum *>(buffer), static_cast<const TWeight *>(weights), ld_weight_col, ld_weight_row);
  }

  using DepthwiseDepthfirstGenericWithMultiplierBase<strategy>::execute;
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
    if (std::numeric_limits<TAccum>::is_integer)
    {
      activation_min = std::numeric_limits<TAccum>::min();
      activation_max = std::numeric_limits<TAccum>::max();
    }
    else
    {
      activation_min = static_cast<TAccum>(-std::numeric_limits<float>::infinity());
      activation_max = static_cast<TAccum>(std::numeric_limits<float>::infinity());
    }

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

    // Get a function to call for each point of the output
    auto tile_fn = [&] (const TInput **inptrs,
                        TOutput **outptrs,
                        const TWeight *weights,
                        const unsigned int,
                        const unsigned int start_output_channel) {
#ifdef CYCLE_PROFILING
      auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(strategy::output_rows() * strategy::output_cols() * this->m_args.channel_multiplier * this->m_args.kernel_rows * this->m_args.kernel_cols));
#endif
      strat.kernel(
        inptrs, outptrs, weights,
        m_biases ? m_biases + start_output_channel : nullptr,
        this->kernel_points(), this->m_args.channel_multiplier,
        activation_min, activation_max
      );
    };

    Parent::execute_tiles(
      tile_fn, 0.0f,
      batches, input_height, input_width, input_channels, padding,
      _input, ld_input_col, ld_input_row, ld_input_batch,
      parameters,
      output_height, output_width,
      _output, ld_output_col, ld_output_row, ld_output_batch,
      _working_space, thread_id, n_threads
    );
  }
};

}  // namespace depthwise
}  // namespace arm_conv
