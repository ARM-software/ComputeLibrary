/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "winograd_implementations.hpp"
#include "weight_transform.hpp"

namespace arm_conv {
namespace winograd {
namespace weight_transform {

#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#endif  // defined(__aarch64__)
void arm_fp32_4x4_3x3(unsigned int, const float *, size_t, size_t, float *, size_t);
void arm_fp32_2x2_3x3(unsigned int, const float *, size_t, size_t, float *, size_t);
void arm_fp32_2x2_5x5(unsigned int, const float *, size_t, size_t, float *, size_t);
void cpp_fp32_1x6_1x3(unsigned int, const float *, size_t, size_t, float *, size_t);
void cpp_fp32_1x4_1x5(unsigned int, const float *, size_t, size_t, float *, size_t);
void cpp_fp32_1x2_1x7(unsigned int, const float *, size_t, size_t, float *, size_t);

#define IMPL(KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN) \
  new Transform<float>(#KERN, KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN)

#define IMPL_T(KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, KERN) \
  new Transform<float>(#KERN, KERN_ROWS, KERN_COLS, TRANS_ROWS, TRANS_COLS, Transform<float>::get_transposed_kernel(KERN))

static const TransformImplementation<float> transforms_fp32[] = {
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#endif  // defined(__aarch64__)
  { IMPL(3, 3, 6, 6, arm_fp32_4x4_3x3) },
  { IMPL(3, 3, 4, 4, arm_fp32_2x2_3x3) },
  { IMPL(5, 5, 6, 6, arm_fp32_2x2_5x5) },
  { IMPL(1, 3, 1, 8, cpp_fp32_1x6_1x3) },
  { IMPL_T(3, 1, 8, 1, cpp_fp32_1x6_1x3) },
  { IMPL(1, 5, 1, 8, cpp_fp32_1x4_1x5) },
  { IMPL_T(5, 1, 8, 1, cpp_fp32_1x4_1x5) },
  { IMPL(1, 7, 1, 8, cpp_fp32_1x2_1x7) },
  { IMPL_T(7, 1, 8, 1, cpp_fp32_1x2_1x7) },
  { nullptr }
};

template <>
const TransformImplementation<float> *implementation_list(void)
{
  return transforms_fp32;
}

}  // namespace weight_transform
}  // namespace winograd
}  // namespace arm_conv
