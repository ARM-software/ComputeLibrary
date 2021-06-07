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
#include "src/core/NEON/kernels/arm_conv/depthwise/interleaves/list.hpp"

#include <cstdint>

#pragma once

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst_impl(const uint8_t *const *, uint8_t *const *, const void *, uint64_t, const arm_gemm::Requantize32&);

struct sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst
{
  typedef uint32_t bias_type;
  typedef uint8_t input_type;
  typedef uint8_t weight_type;
  typedef uint8_t return_type;

  constexpr static arm_gemm::VLType vl_type = arm_gemm::VLType::SVE;

  typedef void (*kern_type)(const uint8_t *const *, uint8_t *const *, const void *, uint64_t, const arm_gemm::Requantize32&);
  typedef void (*parameter_packing_fn)(unsigned int, void *, const int32_t *, const uint8_t *, const arm_gemm::Requantize32 &, size_t, size_t);
  typedef size_t (*parameter_sizing_fn)(const DepthwiseArgs &);

  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 2;

  constexpr static unsigned int input_rows = 4;
  constexpr static unsigned int input_cols = 4;

  constexpr static parameter_packing_fn pack_parameters = interleave_sve_u8q_3x3_dot::pack_parameters;
  constexpr static parameter_sizing_fn get_packed_size = interleave_sve_u8q_3x3_dot::get_packed_size;

  kern_type kernel = sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst_impl;

  sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst(const CPUInfo *) {}
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
