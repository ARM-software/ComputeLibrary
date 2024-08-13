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

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include "output_transform.hpp"
#include "winograd_implementations.hpp"

namespace arm_conv {
namespace winograd {
namespace output_transform {

void a64_fp16_4x4_3x3(unsigned int, const __fp16 *, size_t, const __fp16 *, __fp16 *, size_t, size_t, __fp16, __fp16);

#define IMPL(OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC, DRIVER) \
  new Transform ## DRIVER <__fp16, __fp16>(#FUNC, OUT_HEIGHT, OUT_WIDTH, KERN_HEIGHT, KERN_WIDTH, FUNC)


template <>
const TransformImplementation<__fp16> *implementation_list(void)
{
  static const TransformImplementation<__fp16> transforms_fp16[] = {
    { IMPL(4, 4, 3, 3, a64_fp16_4x4_3x3, Unpadded) },
    { nullptr }
  };
  return transforms_fp16;
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace arm_conv

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
