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

namespace
{

// We have two sets of quantized kernels; those which use the dot-product
// instructions and which require the biases and quantisation parameters to be
// ravelled into weights/parameter array, and those which use the MLAL
// instructions and which consume separate bias and quantisation parameter
// arrays. The following code adapts these two sets of kernels to use the same
// API - allowing the same driver loop to call them both.

template <typename TIn, typename TWeight, typename TOut>
using UnravelledKernFn = std::function<void(unsigned int, const TIn *const *, const TWeight *, const int32_t *, const arm_gemm::Requantize32 &, const int32_t *, const int32_t *, TOut *const *)>;

template <typename TIn, typename TOut>
using RavelledKernFn = std::function<void(const TIn *const *, TOut *const *, const void *, uint64_t, const arm_gemm::Requantize32 &)>;

template <typename TIn, typename TWeight, typename TOut>
const UnravelledKernFn<TIn, TWeight, TOut> get_unified_kernel(const UnravelledKernFn<TIn, TWeight, TOut> &f) { return f; }

template <typename TIn, typename TWeight, typename TOut>
const UnravelledKernFn<TIn, TWeight, TOut> get_unified_kernel(const RavelledKernFn<TIn, TOut> &f)
{
  return [f] (const unsigned int n_channels,
              const TIn *const *const inptrs,
              const TWeight *const weights,
              const int32_t *,  // Bias (ravelled)
              const arm_gemm::Requantize32 &qp,
              const int32_t *,  // Requantisation muls (ravelled)
              const int32_t *,  // Requantisation shifts (ravelled)
              TOut *const *const outptrs) {
    return f(inptrs, outptrs, weights, n_channels, qp);
  };
}

template <typename T>
using UnravelledPackingFn = std::function<void(unsigned int, void *, const T *, size_t, size_t)>;

template <typename T>
using RavelledPackingFn = std::function<void(unsigned int, void *, const int32_t *, const T *, const arm_gemm::Requantize32 &, size_t, size_t)>;

template <typename T>
const RavelledPackingFn<T> get_unified_packer(const UnravelledPackingFn<T> &f)
{
  return [f] (const unsigned int n_channels,
              void *buffer,
              const int32_t *,  // Bias
              const T *weights,
              const arm_gemm::Requantize32 &,
              size_t ld_weight_col,
              size_t ld_weight_row)
  {
    return f(n_channels, buffer, weights, ld_weight_col, ld_weight_row);
  };
}

template <typename T>
const RavelledPackingFn<T> get_unified_packer(const RavelledPackingFn<T> &f) { return f; }

template <typename T>
constexpr bool requires_unravelled_bias_and_quant_params(const UnravelledPackingFn<T> &) { return true; }

template <typename T>
constexpr bool requires_unravelled_bias_and_quant_params(const RavelledPackingFn<T> &) { return false; }

template <class strategy>
constexpr bool strategy_requires_unravelled_bias_and_quant_params(void)
{
  return requires_unravelled_bias_and_quant_params<typename strategy::weight_type>(strategy::pack_parameters);
}

}

template <class strategy>
class DepthwiseDepthfirstQuantized :
  public DepthwiseCommon<typename strategy::input_type,
                         typename strategy::weight_type,
                         typename strategy::return_type>
{
  using TInput = typename strategy::input_type;
  using TWeight = typename strategy::weight_type;
  using TOutput = typename strategy::return_type;
  using TAccum = typename strategy::bias_type;

  arm_gemm::Requantize32 m_qp;

  size_t sizeof_input_buffer(unsigned int n_channels) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TInput>(strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(n_channels, vl);
    return sizeof(TInput) * rounded_channels;
  }

  size_t sizeof_output_buffer(unsigned int n_channels) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<TOutput>(strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(n_channels, vl);
    return sizeof(TOutput) * rounded_channels;
  }

  size_t sizeof_bias_buffer(unsigned int n_channels) const
  {
    if (strategy_requires_unravelled_bias_and_quant_params<strategy>())
    {
      return (m_qp.bias == nullptr) ? sizeof(TAccum) * n_channels : 0;
    }

    return 0;
  }

  size_t sizeof_requant_mul_buffer(unsigned int n_channels) const
  {
    if (strategy_requires_unravelled_bias_and_quant_params<strategy>())
    {
      return m_qp.per_channel_requant ? 0 : sizeof(int32_t) * n_channels;
    }

    return 0;
  }

  size_t sizeof_requant_shift_buffer(unsigned int n_channels) const
  {
    if (strategy_requires_unravelled_bias_and_quant_params<strategy>())
    {
      return m_qp.per_channel_requant ? 0 : sizeof(int32_t) * n_channels;
    }

    return 0;
  }

  public:
  DepthwiseDepthfirstQuantized(const DepthwiseArgs &args, const arm_gemm::Requantize32 &qp)
    : DepthwiseCommon<TInput, TWeight, TOutput>(args), m_qp(qp)
  {
  }

  DepthwiseDepthfirstQuantized(DepthwiseDepthfirstQuantized &) = delete;
  DepthwiseDepthfirstQuantized &operator=(DepthwiseDepthfirstQuantized &) = delete;

  size_t get_storage_size(void) const override
  {
    return strategy::get_packed_size(this->m_args);
  }

  void pack_parameters(void *buffer, const void *const bias, const void *weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    if (strategy_requires_unravelled_bias_and_quant_params<strategy>())
    {
      m_qp.bias = static_cast<const int32_t *>(bias);
    }

    get_unified_packer<TWeight>(strategy::pack_parameters)(
      this->m_args.input_channels,
      buffer,
      static_cast<const int32_t *>(bias),
      reinterpret_cast<const TWeight *>(weights),
      m_qp,
      ld_weight_col,
      ld_weight_row
    );
  }

  size_t get_working_size(const unsigned int n_threads, const unsigned int n_channels) const override
  {
    const unsigned int n_output_channels = n_channels * this->m_args.channel_multiplier;
    return n_threads * (
      sizeof_output_buffer(n_output_channels) +
      sizeof_input_buffer(n_channels) +
      sizeof_bias_buffer(n_channels) +
      sizeof_requant_mul_buffer(n_channels) +
      sizeof_requant_shift_buffer(n_channels)
    );
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
    void *_working_space,
    const unsigned int thread_id,
    const unsigned int n_threads
  ) const override
  {
    strategy strat(this->m_args.cpu_info);
#ifdef CYCLE_PROFILING
    arm_gemm::profiler prof;
#endif
    // Get a unified API for the kernel function
    auto kernel = get_unified_kernel<TInput, TWeight, TOutput>(strat.kernel);

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
    uint8_t *working_space = static_cast<uint8_t *>(_working_space) + get_working_size(thread_id, input_channels);

    TOutput *const output_buffer = reinterpret_cast<TOutput *>(working_space);
    working_space += sizeof_output_buffer(input_channels * this->m_args.channel_multiplier);

    TInput *const input_buffer = reinterpret_cast<TInput *>(working_space);
    working_space += sizeof_input_buffer(input_channels);

    const int32_t *const bias_ptr = (m_qp.bias == nullptr) ? reinterpret_cast<int32_t *>(working_space)
                                                           : m_qp.bias;
    working_space += sizeof_bias_buffer(input_channels * this->m_args.channel_multiplier);

    const int32_t *const requant_mul_vec = !m_qp.per_channel_requant ? reinterpret_cast<int32_t *>(working_space)
                                                                     : m_qp.per_channel_muls;
    working_space += sizeof_requant_mul_buffer(input_channels * this->m_args.channel_multiplier);

    const int32_t *const requant_shift_vec = !m_qp.per_channel_requant ? reinterpret_cast<int32_t *>(working_space)
                                                                       : m_qp.per_channel_right_shifts;

    if (strategy_requires_unravelled_bias_and_quant_params<strategy>())
    {
      // Initialise the bias buffer
      if (m_qp.bias == nullptr)
      {
        for (unsigned int c = 0; c < input_channels * this->m_args.channel_multiplier; c++)
        {
          const_cast<int32_t *>(bias_ptr)[c] = 0;
        }
      }

      // Initialise the requantisation parameters
      if (!m_qp.per_channel_requant)
      {
        for (unsigned int c = 0; c < input_channels * this->m_args.channel_multiplier; c++)
        {
          const_cast<int32_t *>(requant_mul_vec)[c] = m_qp.per_layer_mul;
          const_cast<int32_t *>(requant_shift_vec)[c] = m_qp.per_layer_right_shift;
        }
      }
    }

    // Initialise the input buffer
    for (unsigned int c = 0; c < input_channels; c++)
    {
      input_buffer[c] = static_cast<TInput>(m_qp.a_offset);
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
          auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(strategy::output_rows * strategy::output_cols * this->m_args.kernel_rows * this->m_args.kernel_cols));
#endif
          kernel(
            this->m_args.input_channels,
            inptr_array,
            reinterpret_cast<const TWeight *>(parameters),
            bias_ptr, m_qp, requant_mul_vec, requant_shift_vec,
            outptr_array
          );
        }
      }
    }
  }
};

}  // namespace depthwise
}  // namespace arm_conv
