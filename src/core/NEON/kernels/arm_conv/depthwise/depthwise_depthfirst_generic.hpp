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

template <class Strategy, unsigned OutputRows, unsigned int OutputCols>
class DepthwiseDepthfirstGenericBase :
  public DepthwiseCommon<typename Strategy::input_type,
                         typename Strategy::weight_type,
                         typename Strategy::return_type>
{
  protected:

  using TInput = typename Strategy::input_type;
  using TWeight = typename Strategy::weight_type;
  using TOutput = typename Strategy::return_type;
  using TAccum = typename Strategy::bias_type;

  size_t sizeof_input_ptr_array(void) const
  {
    return sizeof(TInput *) * this->m_args.kernel_rows * this->m_args.kernel_cols * Strategy::n_output_points;
  }

  size_t sizeof_input_buffer(unsigned int n_channels) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TInput>(Strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(n_channels, vl);
    return sizeof(TInput) * rounded_channels;
  }

  size_t sizeof_output_buffer(unsigned int n_channels) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TOutput>(Strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(n_channels, vl);
    return sizeof(TOutput) * rounded_channels;
  }

  unsigned int input_rows(void) const
  {
    return this->m_args.kernel_rows + (OutputRows - 1)*this->m_args.stride_rows;
  }

  unsigned int input_cols(void) const
  {
    return this->m_args.kernel_cols + (OutputCols - 1)*this->m_args.stride_cols;
  }

  void execute_tiles(
    std::function<void(const TInput *const *, TOutput *const *)> tile_fn,
    std::function<void(TInput *, unsigned int)> initialise_input_buffer,
    const unsigned int batches,
    const unsigned int input_height,
    const unsigned int input_width,
    const unsigned int input_channels,
    const PaddingValues &padding,
    const void *const _input,
    const size_t ld_input_col,
    const size_t ld_input_row,
    const size_t ld_input_batch,
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
    static_assert(OutputRows * OutputCols <= Strategy::n_output_points,
                  "Too many output points for kernel.");

    // Determine what portion of the work to do.
    const unsigned int n_rows_per_thread = arm_gemm::iceildiv(output_height, n_threads);
    const int start_out_height = std::min(thread_id * n_rows_per_thread, output_height);
    const int end_out_height = std::min(start_out_height + n_rows_per_thread, output_height);

    // Cast input and output pointers into the right types
    const TInput *const inptr = static_cast<const TInput *>(_input);
    TOutput *const outptr = static_cast<TOutput *>(_output);

    // Allocate portions of the working space
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space) + this->get_working_size(thread_id, input_channels);
    const TInput **const inptr_array = reinterpret_cast<const TInput **>(working_space);
    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space + this->sizeof_input_ptr_array());
    TInput *const input_buffer = reinterpret_cast<TInput *>(working_space + this->sizeof_input_ptr_array() + this->sizeof_output_buffer(input_channels * this->m_args.channel_multiplier));

    // Create an array for the output pointers
    TOutput * _outptr_array[Strategy::n_output_points];
    TOutput **const outptr_array = _outptr_array;

    // Initialise the input buffer
    initialise_input_buffer(input_buffer, input_channels);

    // For each output tile, construct the requisite set of pointers and call
    // into the kernel.
    for (unsigned int batch = 0; batch < batches; batch++)
    {
      // Get batch pointers
      const auto inptr_batch = inptr + batch * ld_input_batch;
      const auto outptr_batch = outptr + batch * ld_output_batch;

      for (int start_out_i = start_out_height;
           start_out_i < end_out_height;
           start_out_i += static_cast<int>(OutputRows))
      {
        const int end_out_i = std::min(start_out_i + OutputRows,
                                       output_height);

        for (int start_out_j = 0;
             start_out_j < static_cast<int>(output_width);
             start_out_j += static_cast<int>(OutputCols))
        {
          const int end_out_j = std::min(start_out_j + OutputCols,
                                         output_width);

          // Fill the pointer arrays with pointers to the input/output buffers.
          for (auto index = 0u;
               index < (Strategy::n_output_points * this->m_args.kernel_rows * this->m_args.kernel_cols);
               index++)
          {
            inptr_array[index] = input_buffer;
          }
          for (auto index = 0u; index < Strategy::n_output_points; index++)
          {
            outptr_array[index] = output_buffer;
          }

          // Construct the pointer arrays together. Note that the input pointer
          // array is striped. Since the array has already been filled with
          // pointers to the padding array we merely fill in the valid points
          // as we get to them.
          unsigned int output_index = 0;
          auto outptr_row = outptr_batch + start_out_i * ld_output_row + start_out_j * ld_output_col;
          for (auto out_i = start_out_i; out_i < end_out_i; out_i++)
          {
            auto outptr_col = outptr_row;

            // Compute the padding for this row of tiles.
            const int start_in_i = out_i * this->m_args.stride_rows - padding.top;
            const int end_in_i = start_in_i + this->m_args.kernel_rows;
            const auto pad_top = static_cast<unsigned int>(std::max<int>(0, 0 - start_in_i));
            const auto pad_bottom = static_cast<unsigned int>(std::max<int>(0, end_in_i - input_height));
            const unsigned int valid_rows = this->m_args.kernel_rows - pad_top - pad_bottom;

            for (auto out_j = start_out_j; out_j < end_out_j; out_j++, output_index++)
            {
              // Compute the output pointer.
              outptr_array[output_index] = outptr_col;
              outptr_col += ld_output_col;

              // Compute the padding for this tile.
              const int start_in_j = out_j * this->m_args.stride_cols - padding.left;
              const int end_in_j = start_in_j + this->m_args.kernel_cols;
              const auto pad_left = static_cast<unsigned int>(std::max<int>(0, 0 - start_in_j));
              const auto pad_right = static_cast<unsigned int>(std::max<int>(0, end_in_j - input_width));
              const unsigned int valid_cols = this->m_args.kernel_cols - pad_left - pad_right;

              // Hence compute the input pointers.
              auto input_index = output_index + Strategy::n_output_points * (pad_top * this->m_args.kernel_cols + pad_left);
              auto inptr_row = inptr_batch + (start_in_i + pad_top) * ld_input_row + (start_in_j + pad_left) * ld_input_col;
              for (auto in_i = 0u; in_i < valid_rows; in_i++)
              {
                auto inptr_col = inptr_row;
                auto input_index_col = input_index;

                for (auto in_j = 0u; in_j < valid_cols; in_j++)
                {
                  inptr_array[input_index_col] = inptr_col;
                  inptr_col += ld_input_col;
                  input_index_col += Strategy::n_output_points;
                }

                inptr_row += ld_input_row;
                input_index += Strategy::n_output_points * this->m_args.kernel_cols;
              }
            }

            outptr_row += ld_output_row;
          }

          tile_fn(inptr_array, outptr_array);
        }
      }
    }
  }

  public:
  DepthwiseDepthfirstGenericBase(const DepthwiseArgs &args) : DepthwiseCommon<TInput, TWeight, TOutput>(args)
  {
  }

  DepthwiseDepthfirstGenericBase(DepthwiseDepthfirstGenericBase &) = delete;
  DepthwiseDepthfirstGenericBase &operator=(DepthwiseDepthfirstGenericBase &) = delete;

  size_t get_storage_size(void) const override
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TAccum>(Strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(this->m_args.input_channels, vl);
    return (this->m_args.kernel_rows * this->m_args.kernel_cols) * rounded_channels * sizeof(TWeight);
  }

  void pack_parameters(void *_buffer, const void *, const void *_weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    // Cast the pointers
    TWeight *buffer = static_cast<TWeight *>(_buffer);
    const TWeight *const weights = static_cast<const TWeight *>(_weights);

    const unsigned int vl = arm_gemm::utils::get_vector_length<TAccum>(Strategy::vl_type);
    ld_weight_col = (ld_weight_col == 0) ? this->m_args.input_channels : ld_weight_col;
    ld_weight_row = (ld_weight_row == 0) ? this->m_args.kernel_cols * ld_weight_col : ld_weight_row;

    for (unsigned int n = 0; n < this->m_args.input_channels; n += vl)
    {
      const unsigned int todo = std::min(vl, this->m_args.input_channels - n);

      // Copy each of the weights in turn
      auto weights_row = weights + n;
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

  size_t get_working_size(const unsigned int n_threads, const unsigned int n_channels) const override
  {
    const unsigned int n_output_channels = n_channels * this->m_args.channel_multiplier;
    return n_threads * (sizeof_input_ptr_array() +
                        sizeof_output_buffer(n_output_channels) +
                        sizeof_input_buffer(n_channels));
  }
};

template <class Strategy, unsigned OutputRows, unsigned int OutputCols>
class DepthwiseDepthfirstGeneric : public DepthwiseDepthfirstGenericBase<Strategy, OutputRows, OutputCols>
{
  using Parent = DepthwiseDepthfirstGenericBase<Strategy, OutputRows, OutputCols>;
  using TInput = typename Parent::TInput;
  using TWeight = typename Parent::TWeight;
  using TAccum = typename Parent::TAccum;
  using TOutput = typename Parent::TOutput;

  const TAccum *m_bias = nullptr;

  public:
  DepthwiseDepthfirstGeneric(const DepthwiseArgs &args) : Parent(args)
  {
  }

  DepthwiseDepthfirstGeneric(DepthwiseDepthfirstGeneric &) = delete;
  DepthwiseDepthfirstGeneric &operator=(DepthwiseDepthfirstGeneric &) = delete;

  void pack_parameters(void *buffer, const void *bias, const void *weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    m_bias = static_cast<const TAccum *>(bias);
    Parent::pack_parameters(buffer, bias, weights, ld_weight_col, ld_weight_row);
  }

  using DepthwiseDepthfirstGenericBase<Strategy, OutputRows, OutputCols>::execute;
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
    Strategy strat(this->m_args.cpu_info);
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

    // Create a function to initialise the input buffer
    const auto initialise_input_buffer = [] (TInput *const buffer, const unsigned int n) {
      std::memset(buffer, 0, n * sizeof(TInput));
    };

    // Create a function to execute a tile of work
    const auto tile_fn = [&] (const TInput *const *const inptrs, TOutput *const * const outptrs) {
#ifdef CYCLE_PROFILING
      auto p = prof.ScopedProfiler(
        PROFILE_KERNEL,
        (unsigned long) (OutputRows * OutputCols * this->m_args.kernel_rows* this->m_args.kernel_cols)
      );
#endif
      strat.kernel(inptrs, outptrs, parameters, m_bias,
                   this->m_args.kernel_rows * this->m_args.kernel_cols,
                   this->m_args.input_channels, activation_min, activation_max);
    };

    // Call into a parent utility function to do the actual work.
    Parent::execute_tiles(
      tile_fn, initialise_input_buffer,
      batches, input_height, input_width, input_channels, padding,
      _input, ld_input_col, ld_input_row, ld_input_batch,
      output_height, output_width,
      _output, ld_output_col, ld_output_row, ld_output_batch,
      _working_space, thread_id, n_threads
    );
  }
};

}  // namespace depthwise
}  // namespace arm_conv
