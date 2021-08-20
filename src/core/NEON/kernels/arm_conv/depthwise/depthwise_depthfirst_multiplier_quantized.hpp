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

#include "depthwise_depthfirst_multiplier.hpp"

namespace arm_conv {
namespace depthwise {

template <class strategy>
class DepthwiseDepthfirstWithMultiplierQuantized :
  public DepthwiseCommon<typename strategy::input_type,
                         typename strategy::weight_type,
                         typename strategy::return_type>
{
  using Parent = DepthwiseCommon<typename strategy::input_type,
                                 typename strategy::weight_type,
                                 typename strategy::return_type>;
  using TInput = typename strategy::input_type;
  using TWeight = typename strategy::weight_type;
  using TOutput = typename strategy::return_type;

  const arm_gemm::Requantize32 m_qp;

  size_t sizeof_output_buffer(unsigned int n_channels) const
  {
    const unsigned int vl = arm_gemm::utils::get_vector_length<typename strategy::return_type>(strategy::vl_type);
    const auto rounded_channels = arm_gemm::roundup(n_channels, vl);
    return sizeof(typename strategy::return_type) * rounded_channels;
  }

  public:
  DepthwiseDepthfirstWithMultiplierQuantized(const DepthwiseArgs &args, const arm_gemm::Requantize32 &qp)
    : Parent(args), m_qp(qp)
  {
  }

  DepthwiseDepthfirstWithMultiplierQuantized(DepthwiseDepthfirstWithMultiplierQuantized &) = delete;
  DepthwiseDepthfirstWithMultiplierQuantized &operator=(DepthwiseDepthfirstWithMultiplierQuantized &) = delete;

  size_t get_storage_size(void) const override
  {
    // We produce VL<int32_t> channels at a time, for each of these blocks of
    // channels we store a vector of biases, weights (complicated) and
    // requantize parameters.
    const unsigned int iter_length =
      arm_gemm::utils::get_vector_length<int32_t>(strategy::vl_type);
    const unsigned int n_iters =
      this->m_args.input_channels * arm_gemm::iceildiv(this->m_args.channel_multiplier, iter_length);

    // Compute the cost of storing the weights
    const unsigned int n_dots_per_kernel_row = arm_gemm::iceildiv(strategy::kernel_cols, 4u);

    return n_iters * iter_length * (
        sizeof(int32_t) +  // Bias
        4 * n_dots_per_kernel_row * strategy::kernel_rows * sizeof(TWeight) +  // Weights
        2 * sizeof(int32_t)  // Requantisation parameters
    );
  }

  // We'll want an optimised version of this, but for now a C++ implementation
  // is probably sufficient.
  void pack_parameters(void *_buffer, const void *_biases, const void *_weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    auto buffer = static_cast<uint8_t *>(_buffer);
    auto biases = static_cast<const int32_t *>(_biases);
    auto weights = static_cast<const TWeight *>(_weights);
    auto requant_muls = m_qp.per_channel_muls;
    auto requant_shifts = m_qp.per_channel_right_shifts;

    const unsigned int iter_length =
      arm_gemm::utils::get_vector_length<int32_t>(strategy::vl_type);
    const unsigned int n_iters_per_input_channel =
      arm_gemm::iceildiv(this->m_args.channel_multiplier, iter_length);

    const unsigned int n_dots_per_kernel_row = arm_gemm::iceildiv(strategy::kernel_cols, 4u);

    const size_t iter_stride = iter_length * (
        sizeof(int32_t) +  // Bias
        4 * n_dots_per_kernel_row * strategy::kernel_rows * sizeof(int8_t) +  // Weights
        2 * sizeof(int32_t)  // Requantisation parameters
    );

    ld_weight_col = (ld_weight_col == 0) ? this->m_args.input_channels * this->m_args.channel_multiplier : ld_weight_col;
    ld_weight_row = (ld_weight_row == 0) ? this->m_args.kernel_cols * ld_weight_col : ld_weight_row;

    for (unsigned int input_channel = 0; input_channel < this->m_args.input_channels; input_channel++)
    {
      auto buffer_input_channel = buffer + input_channel * n_iters_per_input_channel * iter_stride;
      auto weights_input_channel = weights + input_channel * this->m_args.channel_multiplier;

      for (unsigned int iter = 0; iter < n_iters_per_input_channel; iter++)
      {
        // Get a pointer to the start of this portion of the buffer; consequently
        // derive pointers to the bias, weight and requantisation portions of
        // this frame.
        auto buffer_base = buffer_input_channel + iter_stride * iter;
        auto buffer_biases = reinterpret_cast<int32_t *>(buffer_base);
        auto buffer_weights = buffer_base + sizeof(int32_t) * iter_length;
        auto buffer_requant_mul = reinterpret_cast<int32_t *>(
          buffer_weights + strategy::kernel_rows * n_dots_per_kernel_row * 4 * iter_length);
        auto buffer_requant_shift = buffer_requant_mul + iter_length;
        auto weights_base = weights_input_channel + iter * iter_length;

        // Hence work through the data for this iteration, on a
        // channel-by-channel basis.
        const auto this_iter_length = std::min<unsigned int>(
          iter_length, this->m_args.channel_multiplier - iter * iter_length
        );
        for (unsigned int i = 0; i < this_iter_length; i++)
        {
          auto weights_channel = weights_base + i;

          // Read the bias value, we modify this as we read the weights.
          auto bias_value = biases == nullptr ? 0 : *(biases++);
          int32_t elements_sum = 0;

          // Read through the kernel; for each row, marshal together as many dot
          // product terms as are required.
          for (unsigned int ki = 0; ki < strategy::kernel_rows; ki++)
          {
            auto buffer_row = buffer_weights + i*4 + ki * 4 * n_dots_per_kernel_row * iter_length;
            auto weights_row = weights_channel + ki * ld_weight_row;

            unsigned int kj = 0;
            for (; kj < strategy::kernel_cols; kj++)
            {
              // Determine which element to which we're writing
              const auto dot = kj / 4;
              const auto elem = kj % 4;

              // Copy the value; include in the sum
              const auto val = weights_row[kj * ld_weight_col];
              buffer_row[dot * 4 * iter_length + elem] = val;
              elements_sum += val;
            }
            for (; kj < 4 * n_dots_per_kernel_row; kj++)
            {
              const auto dot = kj / 4;
              const auto elem = kj % 4;
              buffer_row[dot * 4 * iter_length + elem] = 0;
            }

            buffer_row += 4 * n_dots_per_kernel_row * iter_length;
          }

          // Write back the bias and offset values
          *(buffer_biases++) =
            bias_value - m_qp.a_offset * elements_sum +
            strategy::kernel_rows * strategy::kernel_cols * m_qp.a_offset * m_qp.b_offset;

          // Write out the requantisation parameters
          *(buffer_requant_mul++) = m_qp.per_channel_requant ? *(requant_muls++) : m_qp.per_layer_mul;
          *(buffer_requant_shift++) = m_qp.per_channel_requant ? *(requant_shifts++) : m_qp.per_layer_right_shift;
        }
      }
    }
  }

  size_t get_working_size(const unsigned int n_threads, const unsigned int n_channels) const override
  {
    const unsigned int n_output_channels = n_channels * this->m_args.channel_multiplier;
    return n_threads * sizeof_output_buffer(n_output_channels);
  }

  using Parent::execute;
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

    auto executefn = [strat, this] (
      const TInput *const *const inptrs,
      TOutput *const *const outptr_array,
      const void *const params
    ) {
      strat.kernel(inptrs, outptr_array, params, this->m_args.channel_multiplier, m_qp);
    };

    // Get working space for this thread
    uint8_t *const working_space = static_cast<uint8_t *>(_working_space) + get_working_size(1, input_channels) * thread_id;

    // Determine the stride across blocks of parameters
    const unsigned int iter_length =
      arm_gemm::utils::get_vector_length<int32_t>(strategy::vl_type);
    const unsigned int n_iters_per_input_channel = arm_gemm::iceildiv(this->m_args.channel_multiplier, iter_length);
    const unsigned int n_dots_per_kernel_row = arm_gemm::iceildiv(strategy::kernel_cols, 4u);
    const size_t param_stride = n_iters_per_input_channel * iter_length * (
        sizeof(int32_t) +  // Bias
        4 * n_dots_per_kernel_row * strategy::kernel_rows * sizeof(int8_t) +  // Weights
        2 * sizeof(int32_t)  // Requantisation parameters
    );

    common::depthwise_multiplier_execute<strategy>(
      executefn, m_qp.a_offset, this->m_args,
      batches, input_height, input_width, input_channels, padding,
      _input, ld_input_col, ld_input_row, ld_input_batch,
      parameters, param_stride,
      output_height, output_width,
      _output, ld_output_col, ld_output_row, ld_output_batch,
      working_space, thread_id, n_threads
    );
  }
};

}  // namespace depthwise
}  // namespace arm_conv
