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

#include <cstdint>

#pragma once

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace pooling {

void sve_fp16_nhwc_avg_generic_depthfirst_impl(const uint64_t window_cells, const uint64_t n_valid_cells, uint64_t n_channels, const __fp16 *const *const inptrs, __fp16 *outptr);

struct sve_fp16_nhwc_avg_generic_depthfirst
{
  typedef __fp16 operand_type;
  typedef __fp16 return_type;

  typedef void (*kern_type)(const uint64_t window_cells, const uint64_t n_valid_cells, uint64_t n_channels, const __fp16 *const *const inptrs, __fp16 *outptr);

  constexpr static PoolingType pooling_type(void) { return PoolingType::AVERAGE; }


  kern_type kernel = sve_fp16_nhwc_avg_generic_depthfirst_impl;

  sve_fp16_nhwc_avg_generic_depthfirst(const CPUInfo *) {}
};

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE) && defined(__ARM_FP16_ARGS)
