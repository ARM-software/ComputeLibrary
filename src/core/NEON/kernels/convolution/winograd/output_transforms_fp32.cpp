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

#include "output_transform.hpp"
#include "winograd_implementations.hpp"

namespace arm_conv {
namespace winograd {
namespace output_transform {

void arm_fp32_4x4_3x3(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_2x2_3x3(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_2x2_5x5(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_1x6_1x3(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_1x4_1x5(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);
void arm_fp32_1x2_1x7(unsigned int, const float *, size_t, const float *, float *, size_t, size_t, float, float);

#define IMPL(OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC, DRIVER) \
  new Transform ## DRIVER <float, float>(#FUNC, OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC)

#define IMPL_T(OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC, DRIVER) \
  new Transform ## DRIVER <float, float>(#FUNC, OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, Transform ## DRIVER <float, float>::get_transposed_kernel(FUNC))

static const TransformImplementation<float> transforms_fp32[] = {
#if defined(__aarch64__)
#endif  // defined(__aarch64__)
  { IMPL(4, 4, 3, 3, arm_fp32_4x4_3x3, Unpadded), MethodConstraints::LargerShape },
  { IMPL(2, 2, 3, 3, arm_fp32_2x2_3x3, Unpadded) },
  { IMPL(2, 2, 5, 5, arm_fp32_2x2_5x5, Unpadded) },
  { IMPL(1, 6, 1, 3, arm_fp32_1x6_1x3, Unpadded) },
  { IMPL_T(6, 1, 3, 1, arm_fp32_1x6_1x3, Unpadded) },
  { IMPL(1, 4, 1, 5, arm_fp32_1x4_1x5, Unpadded) },
  { IMPL_T(4, 1, 5, 1, arm_fp32_1x4_1x5, Unpadded) },
  { IMPL(1, 2, 1, 7, arm_fp32_1x2_1x7, Unpadded) },
  { IMPL_T(2, 1, 7, 1, arm_fp32_1x2_1x7, Unpadded) },
  { nullptr }
};

template <>
const TransformImplementation<float> *implementation_list(void)
{
  return transforms_fp32;
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace arm_conv
