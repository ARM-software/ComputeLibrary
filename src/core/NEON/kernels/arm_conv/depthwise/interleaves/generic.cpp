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

#include "generic.hpp"

#include <functional>

namespace arm_conv {
namespace depthwise {
namespace interleaves {

PackingArguments::PackingArguments(
  unsigned int kernel_rows, unsigned int kernel_cols, size_t weight_element_size,
  bool include_bias, size_t bias_element_size,
  arm_gemm::VLType vl_type, size_t accumulator_element_size, unsigned int accumulator_depth_vl,
  std::function<bool(unsigned int, unsigned int &, unsigned int &)> get_weight_pos
) : kernel_rows(kernel_rows), kernel_cols(kernel_cols), weight_element_size(weight_element_size),
    include_bias(include_bias), bias_element_size(bias_element_size),
    vl_type(vl_type), accumulator_element_size(accumulator_element_size), accumulator_depth_vl(accumulator_depth_vl),
    get_weight_pos(get_weight_pos)
{
}

size_t get_storage_size_generic(const PackingArguments &packing_args, const DepthwiseArgs &args)
{
  // If the channel multiplier is greater than one, then we treat this as a
  // repeated packing of `channel_multiplier`-sized problems.
  if (args.channel_multiplier > 1)
  {
    DepthwiseArgs args_per_input_channel(args);
    args_per_input_channel.input_channels = args.channel_multiplier;
    args_per_input_channel.channel_multiplier = 1;

    return args.input_channels * get_storage_size_generic(packing_args, args_per_input_channel);
  }

  const unsigned int vl =
    packing_args.accumulator_depth_vl *
    arm_gemm::utils::get_vector_length<uint8_t>(packing_args.vl_type) / packing_args.accumulator_element_size;
  const unsigned int n_packs = arm_gemm::iceildiv(args.input_channels, vl);
  const auto pack_size = (packing_args.include_bias ? packing_args.bias_element_size : 0) +
                         packing_args.kernel_points() * packing_args.weight_element_size;
  return n_packs * pack_size * vl;
}

void pack_parameters_generic(
  const PackingArguments &packing_args,
  const DepthwiseArgs &args,
  void *buffer_raw,
  const void *biases_raw,
  const void *weights_raw,
  size_t ld_weight_col,
  size_t ld_weight_row
)
{
  // Cast the pointers to byte sizes
  auto *buffer = static_cast<uint8_t *>(buffer_raw);
  auto *biases = static_cast<const uint8_t *>(biases_raw);
  auto *weights = static_cast<const uint8_t *>(weights_raw);

  // If the channel multiplier is greater than one, then we treat this as a
  // repeated packing of `channel_multiplier`-sized problems.
  if (args.channel_multiplier > 1)
  {
    // Get a modified copy of the depthwise arguments
    DepthwiseArgs args_per_input_channel(args);
    args_per_input_channel.input_channels = args.channel_multiplier;
    args_per_input_channel.channel_multiplier = 1;

    // Resolve the strides here
    ld_weight_col = ld_weight_col ? ld_weight_col : args.input_channels * args.channel_multiplier;
    ld_weight_row = ld_weight_row ? ld_weight_row : ld_weight_col * packing_args.kernel_cols;

    auto per_input_channel_size = get_storage_size_generic(packing_args, args_per_input_channel);

    for (unsigned int c = 0; c < args.input_channels; c++)
    {
      pack_parameters_generic(
        packing_args, args_per_input_channel, buffer, biases, weights, ld_weight_col, ld_weight_row);

      // Update the pointers
      buffer += per_input_channel_size;
      biases += (biases == nullptr) ? 0 : packing_args.bias_element_size * args.channel_multiplier;
      weights += packing_args.weight_element_size * args.channel_multiplier;
    }
    return;
  }

  // Finalise the weight strides
  ld_weight_col = (ld_weight_col == 0) ? args.input_channels : ld_weight_col;
  ld_weight_row = (ld_weight_row == 0) ? packing_args.kernel_cols * ld_weight_col : ld_weight_row;

  const unsigned int vl =
    packing_args.accumulator_depth_vl *
    arm_gemm::utils::get_vector_length<uint8_t>(packing_args.vl_type) / packing_args.accumulator_element_size;

  for (unsigned int n = 0; n < args.input_channels; n += vl)
  {
    const unsigned int todo = std::min(vl, args.input_channels - n);

    if (packing_args.include_bias)
    {
      if (biases != nullptr)
      {
        memcpy(buffer, biases, todo * packing_args.bias_element_size);
        biases += todo * packing_args.bias_element_size;
      }
      else
      {
        memset(buffer, 0, vl * packing_args.bias_element_size);
      }

      buffer += vl * packing_args.bias_element_size;
    }

    // Copy each of the weights in turn
    unsigned int kx, ky;
    for (int kindex = 0; packing_args.get_weight_pos(kindex, kx, ky); kindex++)
    {
      const auto src_ptr = weights + (kx*ld_weight_row + ky*ld_weight_col + n) * packing_args.weight_element_size;
      memcpy(buffer, src_ptr, todo * packing_args.weight_element_size);
      buffer += vl * packing_args.weight_element_size;
    }
  }
}

}  // namespace interleaves
}  // namespace depthwise
}  // namespace arm_conv
