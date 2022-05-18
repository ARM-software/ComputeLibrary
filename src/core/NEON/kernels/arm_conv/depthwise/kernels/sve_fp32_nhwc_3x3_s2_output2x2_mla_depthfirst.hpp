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

#if defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl(const float *const *const, float *const *const, const void *, unsigned int, const float, const float);
void sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(const unsigned int, const unsigned int, const float *, int64_t, int64_t, float *, int64_t, int64_t, const void *, unsigned int, const float, const float);

class sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst : public DepthwiseDepthfirstStrategy<float, float, float, float>
{
  private:
  using Parent = DepthwiseDepthfirstStrategy<float, float, float, float>;
  Parent::IndirectKernelType m_indirect_kernel = sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl;
  Parent::DirectKernelType m_direct_kernel = sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl;

  public:
  using return_type = float;
  constexpr static auto vl_type = arm_gemm::VLType::SVE;

  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 2;
  constexpr static unsigned int stride_cols = 2;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 2;

  sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst(const CPUInfo *)
  : DepthwiseDepthfirstStrategy<float, float, float, float>(2, 3, 2) {}

  arm_gemm::VLType get_vl_type(void) const override { return vl_type; }

  Parent::IndirectKernelType get_indirect_kernel() const override { return m_indirect_kernel; }
  Parent::DirectKernelType get_direct_kernel() const override { return m_direct_kernel; }
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE)
