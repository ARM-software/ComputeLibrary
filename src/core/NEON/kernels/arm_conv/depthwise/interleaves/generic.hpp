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

#pragma once

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "depthwise.hpp"

#include <functional>

namespace arm_conv {
namespace depthwise {
namespace interleaves {

struct PackingArguments
{
  const unsigned int kernel_rows;
  const unsigned int kernel_cols;
  const size_t weight_element_size;
  const bool include_bias;
  const size_t bias_element_size;
  arm_gemm::VLType vl_type;
  const size_t accumulator_element_size;
  const unsigned int accumulator_depth_vl;
  std::function<bool(unsigned int, unsigned int &, unsigned int &)> get_weight_pos;

  unsigned int kernel_points(void) const { return kernel_cols * kernel_rows; }

  PackingArguments(
    unsigned int kernel_rows,
    unsigned int kernel_cols,
    size_t weight_element_size,
    bool include_bias,
    size_t bias_element_size,
    arm_gemm::VLType vl_type,
    size_t accumulator_element_size,
    unsigned int accumulator_depth_vl,
    std::function<bool(unsigned int, unsigned int &, unsigned int &)> get_weight_pos
  );
};

size_t get_storage_size_generic(
  const PackingArguments &packing_args,
  const DepthwiseArgs &args
);

void pack_parameters_generic(
  const PackingArguments &packing_args,
  const DepthwiseArgs &args,
  void *buffer_raw,
  const void *biases_raw,
  const void *weights_raw,
  size_t ld_weight_col,
  size_t ld_weight_row
);

}  // namespace interleaves
}  // namespace depthwise
}  // namespace arm_conv
