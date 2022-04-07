/*
 * Copyright (c) 2021-2022 Arm Limited.
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

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "src/core/NEON/kernels/arm_conv/depthwise/interleaves/list.hpp"

#include <cstdint>

#pragma once

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst_impl(unsigned int, const int8_t *const *, const int8_t *, const int32_t *, const arm_gemm::Requantize32&, const int32_t *, const int32_t *, int8_t *const *);

class sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst : public DepthwiseDepthfirstStrategy<int8_t, int8_t, int8_t, int32_t>
{
  using Parent = DepthwiseDepthfirstStrategy<int8_t, int8_t, int8_t, int32_t>;

  public:
  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst(const CPUInfo *) : Parent(2, 2, 3, 3, 1, 1) {}

  arm_gemm::VLType get_vl_type(void) const override { return arm_gemm::VLType::SVE; }

  Parent::KernelType kernel = sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    return interleave_sve_s8q_3x3_dot::get_packed_size(args);
  }

  void pack_parameters(
    const DepthwiseArgs &args, void *buffer, const void *biases, const arm_gemm::Requantize32 &qp,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const override
  {
    interleave_sve_s8q_3x3_dot::pack_parameters(
      args.input_channels, buffer, reinterpret_cast<const int32_t *>(biases),
      reinterpret_cast<const int8_t *>(weights), qp, ld_weight_col, ld_weight_row
    );
  }
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
