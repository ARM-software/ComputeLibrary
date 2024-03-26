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

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace depthwise {

void sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl(const __fp16 *const *const input_ptrs, __fp16 *const *const outptrs, const void *params, unsigned int n_channels, const __fp16 activation_min, const __fp16 activation_max);
void sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl(const unsigned int n_tile_rows, const unsigned int n_tile_cols, const __fp16 *inptr, int64_t ld_input_row, int64_t ld_input_col, __fp16 *outptr, int64_t ld_output_row, int64_t ld_output_col, const void *params, unsigned int n_channels, const __fp16 activation_min, const __fp16 activation_max);

class sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst : public DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>
{
  private:
  using Parent = DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>;
  Parent::IndirectKernelType m_indirect_kernel = sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl;
  Parent::DirectKernelType m_direct_kernel = sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl;

  public:
  using return_type = __fp16;
  constexpr static auto vl_type = arm_gemm::VLType::SVE;

  constexpr static unsigned int kernel_rows = 5;
  constexpr static unsigned int kernel_cols = 5;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 2;

  sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst(const CPUInfo *)
  : Parent(output_rows, output_cols, kernel_rows, kernel_cols, stride_rows, stride_cols) {}

  arm_gemm::VLType get_vl_type(void) const override { return vl_type; }

  Parent::IndirectKernelType get_indirect_kernel() const override { return m_indirect_kernel; }
  Parent::DirectKernelType get_direct_kernel() const override { return m_direct_kernel; }
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
