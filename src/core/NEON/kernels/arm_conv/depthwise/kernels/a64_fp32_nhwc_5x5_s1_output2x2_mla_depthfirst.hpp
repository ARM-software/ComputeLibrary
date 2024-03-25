/*
 * Copyright (c) 2021-2023 Arm Limited.
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

#include "utils.hpp"

#include <cstdint>

#pragma once

#if defined(__aarch64__)

namespace arm_conv {
namespace depthwise {

void a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl(const float *const *const input_ptrs, float *const *const outptrs, const void *params, unsigned int n_channels, const float activation_min, const float activation_max);
void a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl(const unsigned int n_tile_rows, const unsigned int n_tile_cols, const float *inptr, int64_t ld_input_row, int64_t ld_input_col, float *outptr, int64_t ld_output_row, int64_t ld_output_col, const void *params, unsigned int n_channels, const float activation_min, const float activation_max);

class a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst : public DepthwiseDepthfirstStrategy<float, float, float, float>
{
  private:
  using Parent = DepthwiseDepthfirstStrategy<float, float, float, float>;
  Parent::IndirectKernelType m_indirect_kernel = a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl;
  Parent::DirectKernelType m_direct_kernel = a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl;

  public:
  using return_type = float;
  constexpr static auto vl_type = arm_gemm::VLType::None;

  constexpr static unsigned int kernel_rows = 5;
  constexpr static unsigned int kernel_cols = 5;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 2;

  a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst(const CPUInfo *)
  : Parent(output_rows, output_cols, kernel_rows, kernel_cols, stride_rows, stride_cols) {}

  arm_gemm::VLType get_vl_type(void) const override { return vl_type; }

  Parent::IndirectKernelType get_indirect_kernel() const override { return m_indirect_kernel; }
  Parent::DirectKernelType get_direct_kernel() const override { return m_direct_kernel; }
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__)
