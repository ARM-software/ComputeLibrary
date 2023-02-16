/*
 * Copyright (c) 2022 Arm Limited.
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

namespace arm_conv {
namespace pooling {

void sme_s8q_nhwc_avg_generic_depthfirst_impl(const uint64_t window_cells, const uint64_t n_valid_cells, uint64_t n_channels, const int8_t *const *const inptrs, int8_t *outptr, const Requantize32 &qp);

struct sme_s8q_nhwc_avg_generic_depthfirst : IGenericDepthfirstStrategy<int8_t, int8_t, Requantize32>
{
  using Parent = IGenericDepthfirstStrategy<int8_t, int8_t, Requantize32>;
  sme_s8q_nhwc_avg_generic_depthfirst(const CPUInfo *) {}
  typename Parent::KernelType get_kernel(void) const override { return sme_s8q_nhwc_avg_generic_depthfirst_impl; }
};

}  // namespace pooling
}  // namespace arm_conv
