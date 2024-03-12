/*
 * Copyright (c) 2023 Arm Limited.
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

#include <cstdint>

#pragma once

namespace arm_conv {
namespace depthwise {

void sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_indirect_impl(const __fp16 *const *const input_ptrs, __fp16 *const *const outptrs, const void *params, unsigned int n_channels, const __fp16 activation_min, const __fp16 activation_max);
void sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_direct_impl(const unsigned int n_tile_rows, const unsigned int n_tile_cols, const __fp16 *inptr, int64_t ld_input_row, int64_t ld_input_col, __fp16 *outptr, int64_t ld_output_row, int64_t ld_output_col, const void *params, unsigned int n_channels, const __fp16 activation_min, const __fp16 activation_max);

class sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst : public DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>
{
  private:
  using Parent = DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>;
  Parent::IndirectKernelType m_indirect_kernel = sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_indirect_impl;
  Parent::DirectKernelType m_direct_kernel = sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_direct_impl;

  public:
  using return_type = __fp16;
  constexpr static auto vl_type = arm_gemm::VLType::SME;

  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 3;
  constexpr static unsigned int output_cols = 3;

  sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst(const CPUInfo *)
  : Parent(output_rows, output_cols, kernel_rows, kernel_cols, stride_rows, stride_cols) {}

  arm_gemm::VLType get_vl_type(void) const override { return vl_type; }

  Parent::IndirectKernelType get_indirect_kernel() const override { return m_indirect_kernel; }
  Parent::DirectKernelType get_direct_kernel() const override { return m_direct_kernel; }
};

}  // namespace depthwise
}  // namespace arm_conv
