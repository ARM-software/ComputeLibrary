/*
 * Copyright (c) 2020-2022 Arm Limited.
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

template <typename T>
void cpp_nhwc_1x1_stride_any_depthfirst_impl(const uint64_t, const uint64_t, uint64_t n_channels, const T *const *const inptrs, T *outptr);

template <typename T>
struct cpp_nhwc_1x1_stride_any_depthfirst : IGenericDepthfirstStrategy<T, T, Nothing>
{
  using Parent = IGenericDepthfirstStrategy<T, T, Nothing>;
  cpp_nhwc_1x1_stride_any_depthfirst(const CPUInfo *) {}
  typename Parent::KernelType get_kernel(void) const override { return cpp_nhwc_1x1_stride_any_depthfirst_impl<T>; }
};

}  // namespace pooling
}  // namespace arm_conv
