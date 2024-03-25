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

#include "input_transform.hpp"
#include "winograd_implementations.hpp"

#include <memory>
#include <string>

namespace arm_conv {
namespace winograd {
namespace input_transform {

void a64_fp16_6x6(unsigned int, const __fp16 *, size_t, size_t, __fp16 *, size_t);

#define IMPL(HEIGHT, WIDTH, FUNC, DRIVER) new Transform ## DRIVER <__fp16, __fp16>(#FUNC, HEIGHT, WIDTH, FUNC)

static const TransformImplementation<__fp16> transforms_fp16[] = {
  { IMPL(6, 6, a64_fp16_6x6, Unpadded) },
  { nullptr },
};

template <>
const TransformImplementation<__fp16> *implementation_list(void)
{
  return transforms_fp16;
}

}  // namespace input_transform
}  // namespace winograd
}  // namespace arm_conv

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
