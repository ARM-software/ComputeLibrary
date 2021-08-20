/*
 * Copyright (c) 2021 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl(const float *const *const, float *const *const, const void *, unsigned int, const float, const float);
void sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(const unsigned int, const unsigned int, const float *, int64_t, int64_t, float *, int64_t, int64_t, const void *, unsigned int, const float, const float);

struct sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst
{
  typedef float bias_type;
  typedef float input_type;
  typedef float weight_type;
  typedef float return_type;

  typedef void (*indirect_kern_type)(const float *const *const, float *const *const, const void *, unsigned int, const float, const float);
  typedef void (*direct_kern_type)(const unsigned int, const unsigned int, const float *, int64_t, int64_t, float *, int64_t, int64_t, const void *, unsigned int, const float, const float);

  constexpr static arm_gemm::VLType vl_type = arm_gemm::VLType::SVE;

  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 2;
  constexpr static unsigned int stride_cols = 2;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 2;

  constexpr static unsigned int input_rows = 5;
  constexpr static unsigned int input_cols = 5;

  indirect_kern_type indirect_kernel = sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl;
  direct_kern_type direct_kernel = sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl;

  sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst(const CPUInfo *) {}
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
