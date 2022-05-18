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

#include <cstdint>

#pragma once

#if defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE) &&  defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace depthwise {

void sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_indirect_impl(const __fp16 *const *const, __fp16 *const *const, const void *, unsigned int, const __fp16, const __fp16);
void sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_direct_impl(const unsigned int, const unsigned int, const __fp16 *, int64_t, int64_t, __fp16 *, int64_t, int64_t, const void *, unsigned int, const __fp16, const __fp16);

class sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst : public DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>
{
  private:
  using Parent = DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>;
  Parent::IndirectKernelType m_indirect_kernel = sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_indirect_impl;
  Parent::DirectKernelType m_direct_kernel = sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_direct_impl;

  public:
  using return_type = __fp16;
  constexpr static auto vl_type = arm_gemm::VLType::SVE;

  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 3;
  constexpr static unsigned int output_cols = 3;

  sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst(const CPUInfo *)
  : DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>(3, 3, 1) {}

  arm_gemm::VLType get_vl_type(void) const override { return vl_type; }

  Parent::IndirectKernelType get_indirect_kernel() const override { return m_indirect_kernel; }
  Parent::DirectKernelType get_direct_kernel() const override { return m_direct_kernel; }
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE) &&  defined(__ARM_FP16_ARGS)
