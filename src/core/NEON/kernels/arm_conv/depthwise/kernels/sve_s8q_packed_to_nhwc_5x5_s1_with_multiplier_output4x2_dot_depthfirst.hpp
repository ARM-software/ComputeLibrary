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

#if defined(__ARM_FEATURE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst_impl(const int8_t *const *const, int8_t *const *const, const void *, unsigned int, const arm_gemm::Requantize32&);

struct sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst
{
  typedef int32_t bias_type;
  typedef int8_t input_type;
  typedef int8_t weight_type;
  typedef int8_t return_type;

  typedef void (*kern_type)(const int8_t *const *const, int8_t *const *const, const void *, unsigned int, const arm_gemm::Requantize32&);

  constexpr static arm_gemm::VLType vl_type = arm_gemm::VLType::SVE;

  constexpr static unsigned int kernel_rows = 5;
  constexpr static unsigned int kernel_cols = 5;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 4;
  constexpr static unsigned int output_cols = 2;

  constexpr static unsigned int input_rows = 8;
  constexpr static unsigned int input_cols = 6;
  constexpr static unsigned int input_col_quads = 1;

  kern_type kernel = sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst_impl;

  sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst(const CPUInfo *) {}
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE)
