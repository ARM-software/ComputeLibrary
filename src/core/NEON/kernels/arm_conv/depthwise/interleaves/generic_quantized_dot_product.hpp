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

#include "generic.hpp"

namespace arm_conv {
namespace depthwise {
namespace interleaves {
namespace quantized {

size_t get_storage_size(
  const DepthwiseArgs &args,
  arm_gemm::VLType vl_type,
  unsigned int accumulator_depth_vl=1
);

template <typename T>
void pack_parameters(
  void *buffer, const int32_t *biases,
  const T *weights, size_t ld_weight_col, size_t ld_weight_row,
  const DepthwiseArgs &args,
  const arm_gemm::Requantize32 &qp,
  arm_gemm::VLType vl_type,
  unsigned int accumulator_depth_vl
);

}  // namespace quantized
}  // namespace interleaves
}  // namespace depthwise
}  // namespace arm_conv
