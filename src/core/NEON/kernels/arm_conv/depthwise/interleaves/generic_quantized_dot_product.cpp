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

#include "generic_quantized_dot_product.hpp"
#include <cstdint>

namespace arm_conv {
namespace depthwise {
namespace interleaves {
namespace quantized {

size_t get_storage_size(
  const DepthwiseArgs &args,
  const arm_gemm::VLType vl_type,
  const unsigned int accumulator_depth_vl
)
{
  // We produce VL<int32_t> channels at a time, for each of these blocks of
  // channels we store a vector of biases, weights (complicated) and
  // requantize parameters.
  const unsigned int iter_length = accumulator_depth_vl * arm_gemm::utils::get_vector_length<int32_t>(vl_type);
  const unsigned int n_iters = args.input_channels * arm_gemm::iceildiv(args.channel_multiplier, iter_length);

  // Compute the cost of storing the weights
  const unsigned int n_dots_per_kernel_row = arm_gemm::iceildiv(args.kernel_cols, 4u);

  return n_iters * iter_length * (
    sizeof(int32_t) +  // Bias
    4 * n_dots_per_kernel_row * args.kernel_rows * sizeof(int8_t) +  // Weights
    2 * sizeof(int32_t)  // Requantisation parameters
  );
}

template <typename T>
void pack_parameters(
  void *_buffer, const int32_t *biases,
  const T *weights, size_t ld_weight_col, size_t ld_weight_row,
  const DepthwiseArgs &args,
  const arm_gemm::Requantize32 &qp,
  const arm_gemm::VLType vl_type,
  const unsigned int accumulator_depth_vl
)
{
  auto buffer = static_cast<uint8_t *>(_buffer);
  auto requant_muls = qp.per_channel_muls;
  auto requant_shifts = qp.per_channel_right_shifts;

  const unsigned int iter_length = accumulator_depth_vl * arm_gemm::utils::get_vector_length<int32_t>(vl_type);
  const unsigned int n_iters_per_input_channel = arm_gemm::iceildiv(args.channel_multiplier, iter_length);
  const unsigned int n_dots_per_kernel_row = arm_gemm::iceildiv(args.kernel_cols, 4u);

  const size_t iter_stride = iter_length * (
      sizeof(int32_t) +  // Bias
      4 * n_dots_per_kernel_row * args.kernel_rows * sizeof(T) +  // Weights
      2 * sizeof(int32_t)  // Requantisation parameters
  );

  ld_weight_col = (ld_weight_col == 0) ? args.input_channels * args.channel_multiplier : ld_weight_col;
  ld_weight_row = (ld_weight_row == 0) ? args.kernel_cols * ld_weight_col : ld_weight_row;

  for (unsigned int input_channel = 0; input_channel < args.input_channels; input_channel++)
  {
    auto buffer_input_channel = buffer + input_channel * n_iters_per_input_channel * iter_stride;
    auto weights_input_channel = weights + input_channel * args.channel_multiplier;

    for (unsigned int iter = 0; iter < n_iters_per_input_channel; iter++)
    {
      // Get a pointer to the start of this portion of the buffer; consequently
      // derive pointers to the bias, weight and requantisation portions of
      // this frame.
      auto buffer_base = buffer_input_channel + iter_stride * iter;
      auto buffer_biases = reinterpret_cast<int32_t *>(buffer_base);
      auto buffer_weights = buffer_base + sizeof(int32_t) * iter_length;
      auto buffer_requant_mul = reinterpret_cast<int32_t *>(
        buffer_weights + args.kernel_rows * n_dots_per_kernel_row * 4 * iter_length);
      auto buffer_requant_shift = buffer_requant_mul + iter_length;
      auto weights_base = weights_input_channel + iter * iter_length;

      // Hence work through the data for this iteration, on a
      // channel-by-channel basis.
      const auto this_iter_length = std::min<unsigned int>(
        iter_length, args.channel_multiplier - iter * iter_length
      );
      for (unsigned int i = 0; i < this_iter_length; i++)
      {
        auto weights_channel = weights_base + i;

        // Read the bias value, we modify this as we read the weights.
        auto bias_value = biases == nullptr ? 0 : *(biases++);
        int32_t elements_sum = 0;

        // Read through the kernel; for each row, marshal together as many dot
        // product terms as are required.
        for (unsigned int ki = 0; ki < args.kernel_rows; ki++)
        {
          auto buffer_row = buffer_weights + i*4 + ki * 4 * n_dots_per_kernel_row * iter_length;
          auto weights_row = weights_channel + ki * ld_weight_row;

          unsigned int kj = 0;
          for (; kj < args.kernel_cols; kj++)
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
          bias_value - qp.a_offset * elements_sum +
          args.kernel_rows * args.kernel_cols * qp.a_offset * qp.b_offset;

        // Write out the requantisation parameters
        *(buffer_requant_mul++) = qp.per_channel_requant ? *(requant_muls++) : qp.per_layer_mul;
        *(buffer_requant_shift++) = qp.per_channel_requant ? *(requant_shifts++) : qp.per_layer_right_shift;
      }
    }
  }
}

template void pack_parameters(void *, const int32_t *, const int8_t *, size_t, size_t, const DepthwiseArgs &, const arm_gemm::Requantize32 &, arm_gemm::VLType, unsigned int);
template void pack_parameters(void *, const int32_t *, const uint8_t *, size_t, size_t, const DepthwiseArgs &, const arm_gemm::Requantize32 &, arm_gemm::VLType, unsigned int);

}  // namespace quantized
}  // namespace interleaves
}  // namespace depthwise
}  // namespace arm_conv
