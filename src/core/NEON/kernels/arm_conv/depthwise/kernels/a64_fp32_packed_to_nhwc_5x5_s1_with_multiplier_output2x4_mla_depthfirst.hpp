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

namespace arm_conv {
namespace depthwise {

void a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst_impl(const float *const *const, float *const *const, const void *, const unsigned int, const float, const float);

struct a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst
{
  typedef float bias_type;
  typedef float input_type;
  typedef float weight_type;
  typedef float return_type;

  typedef void (*kern_type)(const float *const *const, float *const *const, const void *, const unsigned int, const float, const float);

  constexpr static arm_gemm::VLType vl_type = arm_gemm::VLType::None;

  constexpr static unsigned int kernel_rows = 5;
  constexpr static unsigned int kernel_cols = 5;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 4;

  constexpr static unsigned int input_rows = 6;
  constexpr static unsigned int input_cols = 8;
  constexpr static unsigned int input_col_quads = 2;

  kern_type kernel = a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst_impl;

  a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst(const CPUInfo *) {}
};

}  // namespace depthwise
}  // namespace arm_conv
